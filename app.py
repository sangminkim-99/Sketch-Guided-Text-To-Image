import typer
from commands.train_LEP import train_LEP
from commands.demo import demo
from commands.sample import sample


app = typer.Typer()
app.command()(train_LEP)
app.command()(demo)
app.command()(sample)


if __name__ == "__main__":
    app()