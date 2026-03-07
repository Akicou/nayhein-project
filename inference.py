#!/usr/bin/env python3
"""
Inference server for the dual-mode AR + Diffusion model.
Supports OpenAI-compatible API with FastAPI.
"""

import argparse
import time
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
import json

from pretrain import DualModeModel, get_default_tokenizer


# Global model and tokenizer
model = None
tokenizer = None
device = None


def load_model(checkpoint_path: str, config_path: str):
    """Load model from checkpoint."""
    global model, tokenizer, device
    
    # Load config
    config = torch.load(config_path, map_location="cpu")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = DualModeModel(
        vocab_size=config.get("vocab_size", 16000),
        hidden_size=config.get("hidden_size", 256),
        num_layers=config.get("num_layers", 6),
        num_heads=config.get("num_heads", 4),
        head_dim=config.get("head_dim", 64),
        max_seq_len=config.get("max_seq_len", 8192),
        use_flash_attn=False,  # Disable for inference
    )
    
    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = get_default_tokenizer()
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    
    return model, tokenizer


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    mode: str = "ar"  # "ar", "diffusion", or "combined"
    stream: bool = False


class CompletionResponse(BaseModel):
    id: str
    text: str
    ttft: float  # Time to first token (seconds)
    tps: float  # Tokens per second
    total_tokens: int
    finish_reason: str


app = FastAPI(title="Dual-Mode AR+Diffusion Model API")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - model loaded via load_model() before server starts
    yield
    # Shutdown


@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Tokenize input
    input_ids = tokenizer.encode(request.prompt, return_tensors="pt").to(device)
    input_length = input_ids.shape[1]
    
    # Generate
    start_time = time.time()
    first_token_time = None
    
    generated_ids = []
    max_length = min(input_length + request.max_tokens, model.max_seq_len)
    
    with torch.no_grad():
        for i in range(max_length - input_length):
            # Forward pass
            outputs = model(input_ids, use_cache=False, mode=request.mode)
            
            # Get logits
            logits = outputs.get("ar_logits", outputs.get("diffusion_logits"))
            if logits is None:
                raise HTTPException(status_code=500, detail=f"No logits for mode: {request.mode}")
            
            # Get next token
            next_token_logits = logits[0, -1, :] / request.temperature
            
            # Apply top-p sampling
            if request.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > request.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Record first token time
            if first_token_time is None:
                first_token_time = time.time() - start_time
            
            # Stop if EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            generated_ids.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Yield for streaming (if implemented)
    
    # Decode
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Calculate metrics
    total_time = time.time() - start_time
    ttft = first_token_time if first_token_time else 0
    tps = len(generated_ids) / total_time if total_time > 0 else 0
    
    return CompletionResponse(
        id=f"cmpl-{int(time.time()*1000)}",
        text=generated_text,
        ttft=ttft,
        tps=tps,
        total_tokens=len(generated_ids),
        finish_reason="stop"
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "dual-mode-ar-diffusion",
                "object": "model",
                "created": 1700000000,
                "owned_by": "local",
                "permission": [],
                "root": "dual-mode-ar-diffusion",
                "parent": None,
            }
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Run inference server")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt file")
    parser.add_argument("--config", type=str, required=True, help="Path to config .pt file")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    args = parser.parse_args()
    
    # Load model
    global model, tokenizer
    model, tokenizer = load_model(args.checkpoint, args.config)
    
    # Run server
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
