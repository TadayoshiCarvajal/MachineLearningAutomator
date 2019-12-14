from utility import get_time_string
from pipeline_automator import PipelineAutomator
from sys import argv
from time import time
    
if __name__ == '__main__':
    """
    This is the main file of Pipeline Automator. 
    Run this file using the following the command in a terminal/command prompt window:

    python main.py "file_name.csv" description_column label_column
    """
    start = time()
    commandline_arguments = argv[1:]

    if len(commandline_arguments) != 3:
        raise ValueError(
    "Must specify 3 arguments: the file name, the description column name, and the target column name.")
    file_name, feature_column, label_column = commandline_arguments
    parameters = {  'feature_col_name' : feature_column,
                    'label_col_name' : label_column }

    # Initialize the pipeline and display the parameters used...
    pipeline = PipelineAutomator(file_name, parameters)

    # Commence the first run cycle through the pipeline...
    pipeline.display_parameters()
    record_type_classifier = pipeline.generate_model()
    pipeline.display_metadata()

    # Compute and show the run time:
    stop = time()
    time_elapsed = get_time_string(stop - start)
    print('Time elapsed:', time_elapsed)