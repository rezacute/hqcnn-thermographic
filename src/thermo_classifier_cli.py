#!/usr/bin/env python3
"""
Thermo Classifier CLI - Thermal Image Classification Tool
===========================================================
A CLI tool for classifying thermal images using hybrid quantum-classical CNNs.
Supports both Medical and Industrial domain analysis.

Usage:
    thermo-classifier train --domain medical --data path/to/data
    thermo-classifier predict --domain industrial --image path/to/image
    thermo-classifier serve --port 8000
"""

import click
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from thermo_classifier.cli.commands import train, predict, serve, evaluate, export


@click.group()
@click.version_option(version="1.0.0")
@click.pass_context
def cli(ctx):
    """Thermo Classifier - Thermal Image Classification with Quantum-Enhanced CNNs
    
    Supports Medical and Industrial domain analysis for thermographic images.
    """
    ctx.ensure_object(dict)


# Register commands
cli.add_command(train)
cli.add_command(predict)
cli.add_command(serve)
cli.add_command(evaluate)
cli.add_command(export)


if __name__ == "__main__":
    cli()
