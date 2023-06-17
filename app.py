import typer
from commands.train_LEP import train_LEP


app = typer.Typer()
app.command()(train_LEP)


@app.command()
def demo(
    port: int = 7777
):
    pass


if __name__ == "__main__":
    app()