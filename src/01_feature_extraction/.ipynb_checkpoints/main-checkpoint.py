import subprocess
import csv
import os

# Define the base parameters
degenerates = ["remove", "replace"]
viruses = ["HIV"]
k_values_for_mash = ["21", "23"]
k_values_other = ["6", "7"]
r_values = ["128", "256"]
n_values = ["Real", "PP", "JustA"]

# Feature extraction scripts with their specific parameters
feature_extraction_scripts = {
    "feature_ACS.py": [],
    "feature_FCGR.py": ["-r"],
    "feature_GSP.py": ["-n"],
    "feature_mash.py": ["-k"],
    "feature_FFP.py": ["-k"],
    "feature_RTD.py": ["-k"],
    "feature_SWF.py": ["-k"],
}

def build_f_arg(script, parameters):
    """
    Constructs the -f argument for the validate_random_forest.py command.
    The -f argument is the name of the script followed by the parameters used, formatted as:
    "script_name(-param1_value1,-param2_value2,...)".
    
    Parameters:
    - script: The script file name.
    - parameters: A list of tuples, each containing a parameter flag and its value.
    
    Returns:
    - A string formatted according to the specifications.
    """
    base_f_arg = script.replace('.py', '')  # Remove the .py extension from script name
    
    if parameters:
        # Format each parameter as "-flag_value"
        formatted_params = [f"{param[0]}_{param[1]}" for param in parameters]
        # Join all parameters with commas and enclose in parentheses
        params_str = ",".join(formatted_params)
        f_arg = f"{base_f_arg}({params_str})"
    else:
        f_arg = base_f_arg
    
    return f_arg

def run_script_with_parameters(script, parameters):
    for degenerate in degenerates:
        for virus in viruses:
            # Correctly initialize validation_params with base parameters
            validation_params = [("-d", degenerate), ("-v", virus)]
            # Correct the way to construct base_cmd to avoid unwanted spaces
            base_cmd = ["python", f"011_feature_generation/{script}"] + [f"{param[0]}{param[1]}" for param in validation_params]
            
            if script in ["feature_mash.py", "feature_FFP.py", "feature_RTD.py", "feature_SWF.py"]:
                values = k_values_for_mash if script == "feature_mash.py" else k_values_other
                for value in values:
                    specific_params = [(parameters[0], value)] if parameters else []
                    cmd = base_cmd + [f"{param[0]}{param[1]}" for param in specific_params]
                    subprocess.run(cmd)
                    f_arg = build_f_arg(script, validation_params + specific_params)
                    subprocess.run(["python", "012_feature_validation/run_model_validation.py", "-v", "HIV", "-f", f_arg])
                    subprocess.run(["python", "013_feature_evaluation/run_model_prediction.py", "-v", "HIV", "-f", f_arg])
            elif script == "feature_FCGR.py":
                for r_value in r_values:
                    specific_params = [(parameters[0], r_value)]
                    cmd = base_cmd + [f"{param[0]}{param[1]}" for param in specific_params]
                    subprocess.run(cmd)
                    f_arg = build_f_arg(script, validation_params + specific_params)
                    subprocess.run(["python", "012_feature_validation/run_model_validation.py", "-v", "HIV", "-f", f_arg])
                    subprocess.run(["python", "013_feature_evaluation/run_model_prediction.py", "-v", "HIV", "-f", f_arg])
            elif script == "feature_GSP.py":
                for n_value in n_values:
                    specific_params = [(parameters[0], n_value)]
                    cmd = base_cmd + [f"{param[0]}{param[1]}" for param in specific_params]
                    subprocess.run(cmd)
                    f_arg = build_f_arg(script, validation_params + specific_params)
                    subprocess.run(["python", "012_feature_validation/run_model_validation.py", "-v", "HIV", "-f", f_arg])
                    subprocess.run(["python", "013_feature_evaluation/run_model_prediction.py", "-v", "HIV", "-f", f_arg])
            else:
                # For scripts without specific parameters, only base_cmd is used
                # subprocess.run(base_cmd)
                f_arg = build_f_arg(script, validation_params)
                subprocess.run(["python", "012_feature_validation/run_model_validation.py", "-v", "HIV", "-f", f_arg])
                subprocess.run(["python", "013_feature_evaluation/run_model_prediction.py", "-v", "HIV", "-f", f_arg])

# Main execution loop
for script, params in feature_extraction_scripts.items():
    run_script_with_parameters(script, params)