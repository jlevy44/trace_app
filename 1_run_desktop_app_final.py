import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
import platform
import time

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

def run_application():
    global port, project_dir
    if port is None:
        messagebox.showerror("Error", "Please set the port first!")
        return
    if project_dir is None:
        messagebox.showerror("Error", "Please select the project directory first!")
        return

    low_memory = low_memory_var.get()
    low_memory_option = "-e LOW_MEMORY=1" if low_memory else ""

    if detect_os() == "Windows":
        command = (
            f"powershell -Command \"cd '{project_dir}'; "
            f"docker run --pull always -d -h localhost -p {port}:{port} "
            f"-v '{project_dir}\\workdir:/workdir' "
            f"-v '{project_dir}\\upload_dir:/upload_dir' "
            f"-v '{project_dir}\\tmpdir:/tmpdir' "
            f"-v '{project_dir}:/pwd' -w /workdir --rm {low_memory_option} "
            f"joshualevy44/trace_app:latest sh -c 'cd /workdir && trace --port {port}'\""
        )
    elif detect_os() in ["Darwin", "Linux"]:
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

    try:
        subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
        time.sleep(5)
        url = f"http://127.0.0.1:{port}/"
        root.clipboard_clear()
        root.clipboard_append(url)
        root.update()
        messagebox.showinfo("Success", f"TRACE running on {url} (copied to clipboard)")
        
        # Open the default browser with the URL
        import webbrowser
        time.sleep(1)
        webbrowser.open(url)
        
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
for i in range(3):
    if i<2: root.columnconfigure(i, weight=1)
    root.rowconfigure(i, weight=1)

# Create and place the buttons
generate_button = tk.Button(root, text="1. Set Project Folder", command=generate_folders)
generate_button.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

run_button = tk.Button(root, text="2. Run TRACE", command=run_application)
run_button.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

quit_button = tk.Button(root, text="3. Quit TRACE", command=quit_application)
quit_button.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

# Create and place the options label
options_label = tk.Label(root, text="Options:")
options_label.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

# Create and place the checkbox for low memory option
low_memory_var = tk.BooleanVar()
low_memory_checkbox = tk.Checkbutton(root, text="Low Memory", variable=low_memory_var)
low_memory_checkbox.grid(row=2, column=0, padx=10, pady=10, sticky="e")

# Create and place the label and text box for setting the port
port_label = tk.Label(root, text="Set Port:")
port_label.grid(row=2, column=1, padx=10, pady=10, sticky="e")

port_entry = tk.Entry(root)
port_entry.grid(row=2, column=2, padx=10, pady=10, sticky="w")
port_entry.insert(0, "8888")
port_entry.bind("<FocusOut>", lambda event: update_port())

if __name__ == "__main__":
    root.mainloop()