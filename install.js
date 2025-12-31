module.exports = {
  run: [
    // Install all dependencies from requirements.txt first
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "uv pip install -r requirements.txt"
        ],
      }
    },
    // Pre-download the Sana model (optional - will download automatically on first use)
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "huggingface-cli download Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"
        ],
      }
    },
    // Install torch with CUDA support at the end (includes xformers and triton)
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          xformers: true,
          triton: true
        }
      }
    }
  ]
}

