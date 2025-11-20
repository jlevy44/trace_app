import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import subprocess
import os
import platform
import time
import pysnooper

image_file_extensions = [".tif", ".tiff", ".ome.tif", ".ome.tiff", ".dng", ".zif", ".stk", ".lsm", ".sgi", ".rgb", ".rgba", ".bw", ".img", ".oif", ".oib", ".sis", ".gel", ".svs", ".scn", ".bif", ".qptiff", ".qpi", ".pki", ".ndpi", ".avs"]
image_file_extensions += list(map(lambda x: x.upper(), image_file_extensions))
image_file_extensions = list(set(image_file_extensions))

def detect_os():
    return platform.system()

def generate_folders():
    global project_dir
    project_dir = filedialog.askdirectory()
    if project_dir:
        os.makedirs(os.path.join(project_dir, 'workdir'), exist_ok=True)
        os.makedirs(os.path.join(project_dir, 'workdir/data'), exist_ok=True)
        os.makedirs(os.path.join(project_dir, 'upload_dir'), exist_ok=True)
        os.makedirs(os.path.join(project_dir, 'tmpdir'), exist_ok=True)
        os.chdir(project_dir)
        messagebox.showinfo("Success", "Folders created successfully!")

def update_port():
    global port
    port = port_entry.get()

def has_xlsx_and_image(upload_dir, image_extensions):
    has_xlsx = False
    has_image = False
    try:
        if not os.path.exists(upload_dir) or not os.path.isdir(upload_dir):
            return False
        for fname in os.listdir(upload_dir):
            path = os.path.join(upload_dir, fname)
            if not os.path.isfile(path):
                continue
            if fname.lower().endswith(".xlsx"):
                has_xlsx = True
            ext = os.path.splitext(fname)[1]
            if ext in image_extensions:
                has_image = True
            if has_xlsx and has_image:
                return True
    except Exception as e:
        return False
    return False

def run_application():
    global port, project_dir
    if port is None:
        messagebox.showerror("Error", "Please set the port first!")
        return
    if project_dir is None:
        messagebox.showerror("Error", "Please select the project directory first!")
        return

    upload_dir = os.path.join(project_dir, 'upload_dir')
    if not has_xlsx_and_image(upload_dir, image_file_extensions):
        messagebox.showerror("Error", "Ensure at least one XLSX and one image file are in the upload_dir before running.")
        return

    low_memory = low_memory_var.get()

    # Get the environment selection
    env_mode = env_mode_var.get()
    conda_env = conda_env_entry.get().strip()
    current_os = detect_os()

    if env_mode == "Current Shell":
        if current_os == "Windows":
            # For Windows, set env var and run with cmd, add LOW_MEMORY if set
            low_memory_cmd = f"set LOW_MEMORY=1&& " if low_memory else ""
            command = (
                f'cd /d "{project_dir}" && '
                f'set PROJECT_DIR={project_dir}&& '
                f'{low_memory_cmd}'
                f'trace --port {port}'
            )
        else:
            # Linux or macOS
            low_memory_cmd = f"export LOW_MEMORY=1 && " if low_memory else ""
            command = (
                f"cd \"{project_dir}\" && "
                f"export PROJECT_DIR=\"{project_dir}\" && "
                f"{low_memory_cmd}"
                f"trace --port {port}"
            )
    elif env_mode == "Conda":
        if not conda_env:
            messagebox.showerror("Error", "Please enter the Conda environment name.")
            return
        if current_os == "Windows":
            # Use cmd call for activating conda in Windows, add LOW_MEMORY if set
            low_memory_cmd = f"set LOW_MEMORY=1&& " if low_memory else ""
            command = (
                f'cd /d "{project_dir}" && '
                f'call conda activate {conda_env} && '
                f'set PROJECT_DIR={project_dir}&& '
                f'{low_memory_cmd}'
                f'trace --port {port}'
            )
        else:
            # Linux or macOS
            low_memory_cmd = f"export LOW_MEMORY=1 && " if low_memory else ""
            command = (
                f"cd \"{project_dir}\" && "
                f"export PROJECT_DIR=\"{project_dir}\" && "
                f"{low_memory_cmd}"
                f"conda activate {conda_env} && "
                f"trace --port {port}"
            )
    elif env_mode == "Docker":
        low_memory_option = "-e LOW_MEMORY=1" if low_memory else ""
        if current_os == "Windows":
            command = (
                f"powershell -Command \"cd '{project_dir}'; "
                f"docker run --pull always -d -h localhost -p {port}:{port} "
                f"-v '{project_dir}\\workdir:/workdir' "
                f"-v '{project_dir}\\upload_dir:/upload_dir' "
                f"-v '{project_dir}\\tmpdir:/tmpdir' "
                f"-v '{project_dir}:/pwd' -w /workdir --rm {low_memory_option} "
                f"joshualevy44/trace_app:latest sh -c 'cd /workdir && trace --port {port}'\""
            )
        elif current_os in ["Darwin", "Linux"]:
            command = (
                f"cd \"{project_dir}\" && "
                f"docker run --pull always -d -h localhost -p {port}:{port} "
                f"-v \"{project_dir}/workdir:/workdir\" "
                f"-v \"{project_dir}/upload_dir:/upload_dir\" "
                f"-v \"{project_dir}/tmpdir:/tmpdir\" "
                f"-v \"{project_dir}:/pwd\" -w /workdir --rm {low_memory_option} "
                f"joshualevy44/trace_app:latest sh -c 'cd /workdir && trace --port {port}'"
            )
        else:
            raise Exception("Unsupported operating system")
    else:
        messagebox.showerror("Error", "Unknown environment mode selected.")
        return

    try:
        # Run the command and capture the result
        # For Windows with "Conda" or "Current Shell", need to specify shell=True and use "cmd"
        # Real-time flush of subprocess output
        print_result = True
        if print_result:
            print("=== TRACE Command Output ===")
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True  # so we get strings not bytes
            )

            url = f"http://127.0.0.1:{port}/"
            root.clipboard_clear()
            root.clipboard_append(url)
            root.update()
            messagebox.showinfo("Success", f"TRACE running on {url} (copied to clipboard)")
            
            # Open the default browser with the URL
            import webbrowser
            time.sleep(3)
            webbrowser.open(url)

            print("STDOUT:")
            while True:
                stdout_line = process.stdout.readline()
                if stdout_line == '' and process.poll() is not None:
                    break
                if stdout_line:
                    print(stdout_line, end='', flush=True)
            print("STDERR:")
            # Drain remaining STDOUT (if any) and flush
            for remaining_line in process.stdout:
                print(remaining_line, end='', flush=True)
            for stderr_line in process.stderr:
                print(stderr_line, end='', flush=True)
            print("===========================")
            process.stdout.close()
            process.stderr.close()
            process.wait()
        
        
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run application: {e}")

def quit_application():
    global port
    if port:
        try:
            command = f"docker ps --filter publish={port} -q"
            container_id = os.popen(command).read().strip()
            if container_id:
                os.system(f"docker stop {container_id}")
                messagebox.showinfo("Success", "TRACE has been terminated.")
            else:
                messagebox.showinfo("Error", "No TRACE container found running on the specified port.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to terminate TRACE: {e}")
    else:
        messagebox.showinfo("Error", "No TRACE process is running.")

# Initialize global variables
port = 8888
project_dir = None

# Create the main window
root = tk.Tk()
root.title("TRACE Application")

# Configure the grid
for i in range(4):
    if i < 3:
        root.columnconfigure(i, weight=1)
    root.rowconfigure(i, weight=1)

# Create and place the buttons
generate_button = tk.Button(root, text="1. Set Project Folder", command=generate_folders)
generate_button.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

run_button = tk.Button(root, text="2. Run TRACE", command=run_application)
run_button.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

quit_button = tk.Button(root, text="3. Quit TRACE", command=quit_application)
quit_button.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

# Create and place the options label with low memory and environment mode dropdown
options_label = tk.Label(root, text="Options:")
options_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")

low_memory_var = tk.BooleanVar()
low_memory_checkbox = tk.Checkbutton(root, text="Low Memory", variable=low_memory_var)
low_memory_checkbox.grid(row=1, column=1, padx=10, pady=10, sticky="w")

# Drop-down menu for environment selection
env_modes = ["Current Shell", "Conda", "Docker"]
env_mode_var = tk.StringVar(value=env_modes[0])
def on_env_mode_change(*args):
    selected = env_mode_var.get()
    if selected == "Conda":
        conda_env_label.grid()
        conda_env_entry.grid()
    else:
        conda_env_label.grid_remove()
        conda_env_entry.grid_remove()
env_mode_dropdown = tk.OptionMenu(root, env_mode_var, *env_modes)
env_mode_dropdown.grid(row=1, column=2, padx=2, pady=10, sticky="w")
env_mode_var.trace_add("write", on_env_mode_change)

# Remove old Use Docker/Conda controls from layout (if any, from previous codebase)

# Create and place the label and text box for setting the port
port_label = tk.Label(root, text="Set Port:")
port_label.grid(row=3, column=0, padx=10, pady=10, sticky="e")

port_entry = tk.Entry(root)
port_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")
port_entry.insert(0, "8888")
port_entry.bind("<FocusOut>", lambda event: update_port())

# Create and place the label and text box for setting the conda environment (shown only if "Conda" is selected)
conda_env_label = tk.Label(root, text="Conda Env:")
conda_env_label.grid(row=3, column=2, padx=10, pady=10, sticky="e")
conda_env_label.grid_remove()  # Hide the label from the beginning

conda_env_entry = tk.Entry(root)
conda_env_entry.grid(row=3, column=3, padx=10, pady=10, sticky="w")
conda_env_entry.grid_remove()  # Hide the grid element from the beginning

def main():
    root.mainloop()

if __name__ == "__main__":
    main()