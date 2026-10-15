# Contributing to AI-Powered Cybersecurity Threat Detection System

Thank you for your interest in improving this educational project!

## Development Setup

1. **Clone the repository**:
   ```powershell
   git clone https://github.com/your-username/ai-powered-cybersecurity-threat-detection.git
   cd ai-powered-cybersecurity-threat-detection
   ```

2. **Set up a virtual environment (PowerShell)**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements-dev.txt
   ```

4. **Run the automated test suite**:
   ```powershell
   python -m pytest tests -v
   ```

## Contribution Guidelines
- Ensure all new features include unit tests in `tests/`.
- Maintain docstrings and type hints for all public functions.
- Never commit private datasets, large binary captures, or credentials.
- All detection rules must be explainable and include a corresponding severity and category.
