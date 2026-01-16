"""
Logan CLI - Log Analysis Tool

A click-based command-line interface for log analysis, templatization, and anomaly detection.
"""

import os
import sys
import click

import http.server
import socketserver

from logan._version import __version__
from logan.log_diagnosis.models.model_zero_shot_classifer import ZeroShotModels
from logan.log_diagnosis.models import ModelType
from logan.log_diagnosis.utils import prepare_output_dir


class ZeroShotModelType(click.ParamType):
    """
    Custom click parameter type for parsing model names.
    Accepts both ZeroShotModels enum values and custom model name strings.
    """
    name = "model"

    def convert(self, value, param, ctx):
        if value is None:
            return ZeroShotModels.CROSSENCODER
        
        # If already a ZeroShotModels instance, return as-is
        if isinstance(value, ZeroShotModels):
            return value
        
        # Try to match against ZeroShotModels enum by name
        try:
            return ZeroShotModels[value.upper()]
        except (KeyError, AttributeError):
            pass
        
        # Try to match against ZeroShotModels enum by value
        for model in ZeroShotModels:
            if model.value == value:
                return model
        
        # If no enum match, return the string as-is (allows custom model names)
        return value


class ModelTypeChoice(click.ParamType):
    """Custom click parameter type for ModelType enum."""
    name = "model_type"

    def convert(self, value, param, ctx):
        if value is None:
            return ModelType.ZERO_SHOT
        
        if isinstance(value, ModelType):
            return value
        
        try:
            return ModelType(value)
        except ValueError:
            self.fail(
                f"Invalid model type: {value}. Choose from: {', '.join([m.value for m in ModelType])}",
                param,
                ctx,
            )


ZERO_SHOT_MODEL = ZeroShotModelType()
MODEL_TYPE_CHOICE = ModelTypeChoice()


@click.group()
@click.version_option(version=__version__, prog_name="logan")
def cli():
    """
    Logan - Log Analysis Tool
    
    A powerful tool for log preprocessing, templatization, and anomaly detection.
    
    \b
    Example usage:
        logan analyze --input-files file1.log file2.log --output-dir ./output
        logan analyze --input-files /path/to/logs/ --time-range 1-day --output-dir ./results
    """
    pass


@cli.command("analyze")
@click.option(
    "--files", "-f",
    multiple=True,
    required=True,
    type=click.Path(exists=True),
    help="Input files or directories for anomaly report generation. Can be specified multiple times."
)
@click.option(
    "--time-range", "-t",
    type=str,
    default="all-data",
    show_default=True,
    help="Time range for analysis. Options: all-data, 1-day, 2-day, ..., 1-week, 2-week, 1-month"
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    required=True,
    help="Directory where output reports will be stored."
)
@click.option(
    "--debug-mode/--no-debug-mode",
    default=True,
    show_default=True,
    help="Enable debug mode for saving debug files."
)
@click.option(
    "--process-log-files/--no-process-log-files",
    default=True,
    show_default=True,
    help="Flag to indicate if .log files should be processed from directories."
)
@click.option(
    "--process-txt-files/--no-process-txt-files",
    default=False,
    show_default=True,
    help="Flag to indicate if .txt files should be processed from directories."
)
@click.option(
    "--model-type",
    type=MODEL_TYPE_CHOICE,
    default="zero_shot",
    show_default=True,
    help="Type of model to use for anomaly detection. Supported: zero_shot, similarity, custom"
)
@click.option(
    "--model", "-m",
    type=ZERO_SHOT_MODEL,
    default="crossencoder",
    show_default=True,
    help="Model to use for classification. Built-in options: bart, crossencoder. Or specify a custom HuggingFace model name."
)
@click.option(
    "--clean-up",
    is_flag=True,
    default=False,
    help="Clean up the output directory if it already exists."
)
def analyze(files, time_range, output_dir, debug_mode, process_log_files, 
            process_txt_files, model_type, model, clean_up):
    """
    Analyze log files for anomalies.
    
    This command performs three main tasks:
    
    \b
    1. Preprocess log files - clean, format, and temporally sort log data
    2. Generate templates - use Drain3 algorithm to create log templates
    3. Detect anomalies - generate anomaly reports based on processed templates
    
    \b
    Examples:
        # Analyze a single log file
        logan analyze -i server.log -o ./output
        
        # Analyze multiple files with time range
        logan analyze -i logs/app.log -i logs/system.log -t 1-day -o ./results
        
        # Analyze a directory of logs
        logan analyze -i /var/log/myapp/ -o ./analysis --process-txt-files
        
        # Use a specific model
        logan analyze -i logs/ -o ./output --model bart
        
        # Clean existing output and run fresh analysis
        logan analyze -i logs/ -o ./output --clean-up
    """
    # Convert debug_mode to string format expected by internal modules
    debug_mode_str = "true" if debug_mode else "false"
    
    click.echo(click.style("Logan Log Analysis Tool", fg="green", bold=True))
    click.echo("=" * 50)
    
    click.echo(f"\nConfiguration:")
    click.echo(f"  Input files: {list(files)}")
    click.echo(f"  Time range: {time_range}")
    click.echo(f"  Output directory: {output_dir}")
    click.echo(f"  Debug mode: {debug_mode}")
    click.echo(f"  Process .log files: {process_log_files}")
    click.echo(f"  Process .txt files: {process_txt_files}")
    click.echo(f"  Model type: {model_type}")
    click.echo(f"  Model: {model}")
    click.echo(f"  Clean up: {clean_up}")
    click.echo()
    
    # Prepare output directory
    click.echo(click.style("Step 0: Preparing output directory...", fg="cyan"))
    prepare_output_dir(output_dir, clean_up)
    
    # Import here to avoid loading heavy dependencies until needed
    from logan.preprocessing.preprocessing import Preprocessing
    from logan.drain.run_drain import Templatizer
    from logan.log_diagnosis.anomaly import Anomaly
    
    # Step 1: Preprocessing
    click.echo(click.style("\nStep 1: Preprocessing log files...", fg="cyan"))
    preprocessing_obj = Preprocessing(debug_mode_str)
    preprocessing_obj.preprocess(
        list(files),
        time_range,
        output_dir,
        process_log_files,
        process_txt_files
    )
    
    if preprocessing_obj.df is None or len(preprocessing_obj.df) == 0:
        click.echo(click.style("\nError: No log lines could be extracted from the input.", fg="red", bold=True))
        sys.exit(1)
    
    click.echo(click.style(f"  Preprocessed {len(preprocessing_obj.df)} log entries", fg="green"))
    
    # Step 2: Template generation
    click.echo(click.style("\nStep 2: Generating log templates...", fg="cyan"))
    drain_config_path = os.path.join(os.path.dirname(__file__), 'drain', 'drain3.ini')
    templatizer = Templatizer(debug_mode=debug_mode_str, config_path=drain_config_path)
    templatizer.miner(
        preprocessing_obj.df,
        output_dir,
        os.path.join(output_dir, "test_templates", "tm-test.templates.json")
    )
    click.echo(click.style("  Templates generated successfully", fg="green"))
    
    # Step 3: Anomaly detection
    click.echo(click.style("\nStep 3: Detecting anomalies...", fg="cyan"))
    anomaly_obj = Anomaly(debug_mode_str, model_type, model)
    anomaly_obj.get_anomaly_report(
        templatizer.df,
        output_dir
    )
    
    click.echo(click.style("\n" + "=" * 50, fg="green"))
    click.echo(click.style("Analysis complete!", fg="green", bold=True))
    click.echo(f"\nOutput files:")
    click.echo(f"  Anomaly report: {os.path.join(output_dir, 'log_diagnosis', 'anomalies.html')}")
    click.echo(f"  Summary report: {os.path.join(output_dir, 'log_diagnosis', 'summary.html')}")


@cli.command()
@click.option(
    "--port", "-p",
    default=8000,
    type=int,
    help="Port for webserver"
)
@click.option(
    "--dir", "-d",
    required=True,
    help="Directory of the report"
)
def view(port, dir):
    """
    View a Job Report.

    This command starts a simple HTTP server to serve the HTML report output from log analysis.

    Usage example:
        logan view --dir OUTPUT_DIR [--port 8080]

    After running the command, visit http://localhost:PORT/OUTPUT_DIR/log_diagnosis/ in your browser
    to view the anomaly and summary reports.
    """
    Handler = http.server.SimpleHTTPRequestHandler

    with socketserver.TCPServer(("", port), Handler) as httpd:
        try:
            click.echo(f"⚠️  Note: This log analysis report contains AI classified results.\n\t Please make sure to manually review the report before taking any action.\n\n")

            click.echo(f"Serving HTTP on port {port}")
            click.echo(f"Please click on this link to view the report: http://localhost:{port}/{dir}/log_diagnosis/")
        
            httpd.serve_forever() 
        except KeyboardInterrupt:
            click.echo("\nClosing Server.")
            httpd.shutdown()


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()

