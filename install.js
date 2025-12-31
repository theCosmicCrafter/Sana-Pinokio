module.exports = {
  run: [
    // Clone Sana repository (needed for SanaPipeline)
    {
      method: "fs.download",
      params: {
        uri: "https://github.com/NVlabs/Sana.git",
        path: "Sana"
      }
    },
    // Create venv and install dependencies
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "uv pip install -r requirements.txt"
        ],
      }
    },
    // Install torch with CUDA support
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
        }
      }
    },
    // Install Sana package from cloned repo (provides SanaPipeline)
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "Sana",
        message: [
          "uv pip install -e ."
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
    }
  ]
}
