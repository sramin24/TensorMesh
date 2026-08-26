import typer

app = typer.Typer(help="TensorMesh: distributed pipeline-parallel LLM inference.")


@app.command()
def version() -> None:
    """Print the TensorMesh version."""
    typer.echo("tensormesh 0.1.0")


if __name__ == "__main__":
    app()
