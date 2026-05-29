"""CLI Commands for Thermo Classifier"""

import click
import torch
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def commands():
    """Thermo Classifier CLI commands"""
    pass


@commands.command()
@click.option('--domain', type=click.Choice(['medical', 'industrial']), required=True, help='Classification domain')
@click.option('--data', type=click.Path(exists=True), required=True, help='Path to training data')
@click.option('--model', default='hq-cnn', help='Model architecture')
@click.option('--epochs', default=100, help='Number of training epochs')
@click.option('--batch-size', default=32, help='Batch size')
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--quantum/--classical', default=True, help='Use quantum-enhanced model')
@click.option('--qubits', default=8, help='Number of qubits for quantum layer')
@click.option('--output', default='outputs/model.pt', help='Output model path')
@click.option('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
def train(domain, data, model, epochs, batch_size, lr, quantum, qubits, output, device):
    """Train a thermal image classifier"""
    console.print(f"[bold green]Training {model} for {domain} domain[/bold green]")
    console.print(f"Data: {data}")
    console.print(f"Device: {device}")
    console.print(f"Quantum-enhanced: {quantum}")
    
    # Import training module
    from thermo_classifier.training import Trainer
    from thermo_classifier.data import ThermalDataset
    
    # Create dataset
    dataset = ThermalDataset(data_path=data, domain=domain)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        domain=domain,
        quantum=quantum,
        qubits=qubits,
        device=device,
        lr=lr,
        batch_size=batch_size
    )
    
    # Train
    trainer.train(dataset, epochs=epochs)
    
    # Save model
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    trainer.save(output)
    console.print(f"[bold]Model saved to {output}[/bold]")


@commands.command()
@click.option('--domain', type=click.Choice(['medical', 'industrial']), required=True, help='Classification domain')
@click.option('--image', type=click.Path(exists=True), required=True, help='Path to thermal image')
@click.option('--model', type=click.Path(exists=True), required=True, help='Path to trained model')
@click.option('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
@click.option('--output', help='Output JSON path for results')
def predict(domain, image, model, device, output):
    """Predict on a thermal image"""
    console.print(f"[bold blue]Predicting {image}[/bold blue]")
    console.print(f"Domain: {domain}")
    
    from thermo_classifier.inference import Predictor
    
    predictor = Predictor(model_path=model, domain=domain, device=device)
    result = predictor.predict(image)
    
    # Display results
    table = Table(title="Prediction Results")
    table.add_column("Class", style="cyan")
    table.add_column("Confidence", style="green")
    
    for cls, conf in result['predictions']:
        table.add_row(cls, f"{conf:.2%}")
    
    console.print(table)
    
    if output:
        import json
        with open(output, 'w') as f:
            json.dump(result, f, indent=2)
        console.print(f"Results saved to {output}")


@commands.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--model', type=click.Path(exists=True), required=True, help='Path to trained model')
@click.option('--domain', type=click.Choice(['medical', 'industrial']), required=True, help='Classification domain')
def serve(host, port, model, domain):
    """Start REST API server"""
    import uvicorn
    from thermo_classifier.api import app
    
    app.state.model_path = model
    app.state.domain = domain
    
    console.print(f"[bold]Starting API server on {host}:{port}[/bold]")
    uvicorn.run(app, host=host, port=port)


@commands.command()
@click.option('--domain', type=click.Choice(['medical', 'industrial']), required=True, help='Classification domain')
@click.option('--model', type=click.Path(exists=True), required=True, help='Path to trained model')
@click.option('--data', type=click.Path(exists=True), required=True, help='Path to test data')
@click.option('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
def evaluate(domain, model, data, device):
    """Evaluate model on test data"""
    console.print(f"[bold yellow]Evaluating {model} on {data}[/bold yellow]")
    
    from thermo_classifier.evaluation import Evaluator
    
    evaluator = Evaluator(model_path=model, domain=domain, device=device)
    results = evaluator.evaluate(data_path=data)
    
    # Display metrics
    table = Table(title="Evaluation Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for metric, value in results.items():
        table.add_row(metric, f"{value:.4f}" if isinstance(value, float) else str(value))
    
    console.print(table)


@commands.command()
@click.option('--model', type=click.Path(exists=True), required=True, help='Path to trained model')
@click.option('--format', type=click.Choice(['onnx', 'torchscript', 'tflite']), default='onnx', help='Export format')
@click.option('--output', required=True, help='Output path')
def export(model, format, output):
    """Export model to different formats"""
    from thermo_classifier.export import export_model
    
    console.print(f"[bold]Exporting {model} to {format}[/bold]")
    export_model(model, output, format=format)
    console.print(f"[bold green]Exported to {output}[/bold green]")
