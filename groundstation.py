import tkinter as tk
from tkinter import ttk, messagebox
import serial
import serial.tools.list_ports
import threading
import time
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting
import numpy as np # For numerical operations, especially vector math
from scipy.spatial.transform import Rotation as R # For quaternion to rotation matrix conversion
import traceback # For detailed error reporting

# --- GLOBALS ---
ser = None # Serial port object
running = False # Flag to control the serial reading thread

# Deque to store the latest quaternion data. Maxlen=1 means it only keeps the very last one.
# SciPy's Rotation.from_quat() expects quaternions in [x, y, z, w] order.
# Initialize with a default quaternion representing no rotation (identity quaternion: x=0, y=0, z=0, w=1)
latest_quaternion = deque([np.array([0, 0, 0, 1])], maxlen=1)

# Deques to store position data for the trail. Maxlen controls the trail length.
# At 50ms update rate, maxlen=500 means 25 seconds of history.
x_positions = deque(maxlen=500)
y_positions = deque(maxlen=500)
z_positions = deque(maxlen=500)

start_time = None # To track the start time of data reception

# Engine states (3 engines: 0, 1, 2)
# Initialize all engines to off (False)
engine_states = [False] * 3 # Changed to 3 engines

# Deque to store raw serial lines
raw_serial_lines = deque(maxlen=200) # Store up to 200 lines for display

# --- GUI SETUP ---
root = tk.Tk()
root.title("RFD900 Attitude and Position Ground Station")
root.configure(bg='#000000')  # Explicitly set root background to DARK_BG_COLOR

# --- Dark Theme Setup ---

style = ttk.Style()
# Use 'clam' as a base theme, as it's good for custom styling
style.theme_use('clam')

# Define dark colors
DARK_BG_COLOR = '#000000'      # Pure black background
MID_DARK_BG_COLOR = '#111111'  # Very dark gray for inputs
VERY_DARK_BORDER = '#222222'   # Subtle dark border
LIGHT_TEXT_COLOR = '#FFFFFF'   # Pure white text
ACCENT_BLUE = '#005288'        # SpaceX signature blue
ACCENT_GREEN = '#00FF00'       # Bright green for indicators
ACCENT_RED = '#FF0000'         # Bright red for alerts
ACCENT_ORANGE = '#FF8800'      # Orange for warnings

# Default font for the application
DEFAULT_FONT = ('Helvetica Neue', 9)

def setup_styles():
    # Configure general widget styles
    style.configure('.', background=DARK_BG_COLOR, foreground=LIGHT_TEXT_COLOR, font=DEFAULT_FONT)
    style.configure('TFrame', background=DARK_BG_COLOR)
    style.configure('TLabel', background=DARK_BG_COLOR, foreground=LIGHT_TEXT_COLOR)

    # Configure TButton to use transparent fill and white outline
    style.configure('TButton',
                    background=DARK_BG_COLOR,
                    foreground=LIGHT_TEXT_COLOR,
                    borderwidth=1,
                    relief='solid')
    style.map('TButton',
              background=[('active', MID_DARK_BG_COLOR), ('pressed', MID_DARK_BG_COLOR)],
              relief=[('pressed', 'flat')])

    # Configure TCombobox to match transparent, white-outlined style
    style.configure('TCombobox',
                    fieldbackground=DARK_BG_COLOR,
                    background=DARK_BG_COLOR,
                    foreground=LIGHT_TEXT_COLOR,
                    insertbackground=LIGHT_TEXT_COLOR,
                    highlightcolor=LIGHT_TEXT_COLOR,
                    borderwidth=1,
                    relief='solid')
    style.map('TCombobox',
              fieldbackground=[('readonly', DARK_BG_COLOR)],
              selectbackground=[('readonly', DARK_BG_COLOR)],
              foreground=[('disabled', '#888888')])

    # Configure TEntry similarly
    style.configure('TEntry',
                    fieldbackground=DARK_BG_COLOR,
                    foreground=LIGHT_TEXT_COLOR,
                    insertbackground=LIGHT_TEXT_COLOR,
                    highlightcolor=DARK_BG_COLOR,
                    borderwidth=1,
                    relief='solid')
    style.map('TEntry',
              fieldbackground=[('disabled', MID_DARK_BG_COLOR)],
              foreground=[('disabled', '#888888')])

    # Configure LabelFrame borders to use the white text color
    style.configure('TLabelframe', background=DARK_BG_COLOR, foreground=LIGHT_TEXT_COLOR, bordercolor=LIGHT_TEXT_COLOR)
    style.configure('TLabelframe.Label', background=DARK_BG_COLOR, foreground=LIGHT_TEXT_COLOR, font=('Helvetica Neue', 12, 'bold'))

    # Define a flat labelframe style with no border (for borderless sections)
    style.configure('Flat.TLabelframe', background=DARK_BG_COLOR, borderwidth=0)
    style.configure('Flat.TLabelframe.Label', background=DARK_BG_COLOR, foreground=LIGHT_TEXT_COLOR, font=('Helvetica Neue', 12, 'bold'))

    # Configure column weights for better resizing in the root window
    root.grid_columnconfigure(0, weight=1) # Left column for telemetry data frames
    root.grid_columnconfigure(1, weight=1) # Right column for plot frames
    root.grid_rowconfigure(2, weight=1) # Row containing telemetry and plots should expand
    root.grid_rowconfigure(4, weight=1) # Allow the raw data section at row 4 to expand

# Apply the styles immediately after style is created
setup_styles()

#
# --- Header/Status Frame ---
status_frame = ttk.Frame(root, style='TFrame')
status_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew", columnspan=2) # Spans both columns
status_frame.grid_columnconfigure(1, weight=1) # Allow time label to expand

date_label = ttk.Label(status_frame, text="Date:--MM-DD")
date_label.grid(row=0, column=0, padx=5, pady=2, sticky="w")

time_label = ttk.Label(status_frame, text="Time: HH:MM:SS")
time_label.grid(row=0, column=1, padx=5, pady=2, sticky="w")

connection_status_label = ttk.Label(status_frame, text="● Disconnected", foreground=ACCENT_RED)
connection_status_label.grid(row=0, column=2, padx=5, pady=2, sticky="e")


#
# --- Serial Configuration Frame ---
config_frame = ttk.LabelFrame(root, text="Serial Configuration", style='TLabelframe')
config_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew", columnspan=2) # Spans both columns
for i in range(6): # Adjusted to 6 columns for connect and disconnect buttons
    config_frame.grid_columnconfigure(i, weight=1) # Make columns in config frame expand

port_label = ttk.Label(config_frame, text="Port:")
port_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")

port_combo = ttk.Combobox(config_frame, values=[], width=20)
port_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

baud_label = ttk.Label(config_frame, text="Baud Rate:")
baud_label.grid(row=0, column=2, padx=5, pady=5, sticky="e")

baud_combo = ttk.Combobox(config_frame, values=["9600", "57600", "115200"], width=8)
baud_combo.current(1) # Default to 57600 baud
baud_combo.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

# Restored connect button
connect_button = ttk.Button(config_frame, text="Connect")
connect_button.grid(row=0, column=4, padx=5, pady=5, sticky="ew")

# Added Disconnect button
disconnect_button = ttk.Button(config_frame, text="Disconnect", command=lambda: disconnect_serial())
disconnect_button.grid(row=0, column=5, padx=5, pady=5, sticky="ew")


#
data_display_frame = ttk.LabelFrame(root, text="Telemetry Display", style='TLabelframe')
data_display_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew") # Placed in left column, spans 1 column
data_display_frame.grid_columnconfigure(0, weight=1) # Allows the first column to expand
data_display_frame.grid_columnconfigure(1, weight=1) # Allows the second column to expand
data_display_frame.grid_rowconfigure(0, weight=0) # For non-expanding elements (timing)
data_display_frame.grid_rowconfigure(1, weight=0) # Spacer row 1
data_display_frame.grid_rowconfigure(2, weight=0) # For non-expanding elements (pyro/battery)
data_display_frame.grid_rowconfigure(3, weight=0) # Spacer row 3
data_display_frame.grid_rowconfigure(4, weight=1) # Row containing system state & engine/raw data (allow these to expand)


font_large = ("Helvetica Neue", 11, "bold")
font_medium = ("Helvetica Neue", 9)
font_small = ("Helvetica", 9) # Slightly smaller font for detailed labels

#
#
# --- Main Timing Header (Flight Time) ---
timing_frame = ttk.LabelFrame(data_display_frame, text="Flight Time", style='TLabelframe')
timing_frame.grid(row=0, column=0, padx=3, pady=3, sticky="ew")
timing_frame.grid_columnconfigure(1, weight=1)

flight_time_label = ttk.Label(timing_frame, text="Flight Time:", font=font_large)
flight_time_label.grid(row=0, column=0, padx=5, pady=2, sticky="w")
flight_time_value_label = ttk.Label(timing_frame, text="00:00:00", font=font_large, foreground=ACCENT_GREEN)
flight_time_value_label.grid(row=0, column=1, padx=5, pady=2, sticky="e")

pyro_frame = ttk.LabelFrame(data_display_frame, text="Pyrotechnics", style='Flat.TLabelframe')
pyro_frame.grid(row=1, column=0, padx=3, pady=3, sticky="ew")
pyro_frame.grid_columnconfigure(1, weight=1)

ttk.Label(pyro_frame, text="Main Chute:", font=font_small).grid(row=0, column=0, padx=5, pady=2, sticky="w")
main_chute_status_label = ttk.Label(pyro_frame, text="READY", font=font_medium, foreground=ACCENT_ORANGE)
main_chute_status_label.grid(row=0, column=1, padx=5, pady=2, sticky="e")

ttk.Label(pyro_frame, text="Drogue Chute:", font=font_small).grid(row=1, column=0, padx=5, pady=2, sticky="w")
drogue_chute_status_label = ttk.Label(pyro_frame, text="READY", font=font_medium, foreground=ACCENT_ORANGE)
drogue_chute_status_label.grid(row=1, column=1, padx=5, pady=2, sticky="e")

ttk.Label(pyro_frame, text="Igniter 1:", font=font_small).grid(row=2, column=0, padx=5, pady=2, sticky="w")
igniter1_status_label = ttk.Label(pyro_frame, text="ARMED", font=font_medium, foreground=ACCENT_RED)
igniter1_status_label.grid(row=2, column=1, padx=5, pady=2, sticky="e")

ttk.Label(pyro_frame, text="Igniter 2:", font=font_small).grid(row=3, column=0, padx=5, pady=2, sticky="w")
igniter2_status_label = ttk.Label(pyro_frame, text="ARMED", font=font_medium, foreground=ACCENT_RED)
igniter2_status_label.grid(row=3, column=1, padx=5, pady=2, sticky="e")

battery_frame = ttk.LabelFrame(data_display_frame, text="Battery", style='Flat.TLabelframe')
battery_frame.grid(row=2, column=0, padx=3, pady=3, sticky="ew")
battery_frame.grid_columnconfigure(1, weight=1)

ttk.Label(battery_frame, text="Voltage:", font=font_small).grid(row=0, column=0, padx=5, pady=2, sticky="w")
battery_voltage_label = ttk.Label(battery_frame, text="12.50 V", font=font_medium, foreground=ACCENT_GREEN)
battery_voltage_label.grid(row=0, column=1, padx=5, pady=2, sticky="e")

ttk.Label(battery_frame, text="Current:", font=font_small).grid(row=1, column=0, padx=5, pady=2, sticky="w")
battery_current_label = ttk.Label(battery_frame, text="0.15 A", font=font_medium, foreground=ACCENT_GREEN)
battery_current_label.grid(row=1, column=1, padx=5, pady=2, sticky="e")

ttk.Label(battery_frame, text="Charge:", font=font_small).grid(row=2, column=0, padx=5, pady=2, sticky="w")
battery_charge_label = ttk.Label(battery_frame, text="95 %", font=font_medium, foreground=ACCENT_GREEN)
battery_charge_label.grid(row=2, column=1, padx=5, pady=2, sticky="e")


# System State Frame
system_state_frame = ttk.LabelFrame(data_display_frame, text="System State", style='Flat.TLabelframe')
system_state_frame.grid(row=3, column=0, padx=3, pady=3, sticky="ew")
system_state_frame.grid_columnconfigure(1, weight=1)
system_state_frame.grid_rowconfigure(2, weight=1) # Allow GPS fix row to expand to push engine/raw data down

ttk.Label(system_state_frame, text="Flight Mode:", font=font_small).grid(row=0, column=0, padx=5, pady=2, sticky="w")
flight_mode_label = ttk.Label(system_state_frame, text="STANDBY", font=font_medium, foreground=ACCENT_BLUE)
flight_mode_label.grid(row=0, column=1, padx=5, pady=2, sticky="e")

ttk.Label(system_state_frame, text="Altitude Source:", font=font_small).grid(row=1, column=0, padx=5, pady=2, sticky="w")
altitude_source_label = ttk.Label(system_state_frame, text="BARO", font=font_medium, foreground=ACCENT_BLUE)
altitude_source_label.grid(row=1, column=1, padx=5, pady=2, sticky="e")

ttk.Label(system_state_frame, text="GPS Fix:", font=font_small).grid(row=2, column=0, padx=5, pady=2, sticky="nw") # sticky nw to keep it top-left
gps_fix_label = ttk.Label(system_state_frame, text="3D Fix (8 Sats)", font=font_medium, foreground=ACCENT_BLUE)
gps_fix_label.grid(row=2, column=1, padx=5, pady=2, sticky="ne") # sticky ne to keep it top-right


# --- Engine Performance Section ---
engine_performance_frame = ttk.LabelFrame(data_display_frame, text="Engine Performance", style='Flat.TLabelframe')
engine_performance_frame.grid(row=4, column=0, padx=3, pady=3, sticky="n")

# Create a single Canvas for the engine diagram, smaller and centered
canvas_width = 260
canvas_height = 220
engine_canvas = tk.Canvas(engine_performance_frame, width=canvas_width, height=canvas_height,
                          bg=DARK_BG_COLOR, highlightthickness=0)
engine_canvas.pack(padx=0, pady=0, anchor="center")

# Compute triangle parameters
r = 30               # Radius of each engine circle
side = 120           # Desired distance between bottom engine centers
cx = canvas_width // 2
cy = r + 15          # Vertical offset for top engine
triangle_height = int((3**0.5) / 2 * side)

# Draw the three engines in an equilateral triangle
engine_canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                          outline=LIGHT_TEXT_COLOR, width=2, fill=DARK_BG_COLOR, tags="eng0")
bl_x = cx - side // 2
bl_y = cy + triangle_height
engine_canvas.create_oval(bl_x - r, bl_y - r, bl_x + r, bl_y + r,
                          outline=LIGHT_TEXT_COLOR, width=2, fill=DARK_BG_COLOR, tags="eng1")
br_x = cx + side // 2
br_y = bl_y
engine_canvas.create_oval(br_x - r, br_y - r, br_x + r, br_y + r,
                          outline=LIGHT_TEXT_COLOR, width=2, fill=DARK_BG_COLOR, tags="eng2")


# --- Engine Metrics Section ---
engine_metrics_frame = ttk.LabelFrame(data_display_frame, text="Engine Metrics", style='Flat.TLabelframe')
engine_metrics_frame.grid(row=5, column=0, padx=3, pady=3, sticky="ew")
# Configure columns: metric label + three engines
for c in range(4):
    engine_metrics_frame.grid_columnconfigure(c, weight=1)

# Header row
ttk.Label(engine_metrics_frame, text="", font=font_small).grid(row=0, column=0, padx=2, pady=2)
for i in range(3):
    ttk.Label(engine_metrics_frame, text=f"Engine {i+1}", font=font_small).grid(row=0, column=i+1, padx=2, pady=2)

# Initialize label lists
servo_angle1_labels = []
servo_angle2_labels = []
throttle_labels = []
thrust_labels = []

# Servo Angle 1
ttk.Label(engine_metrics_frame, text="Servo Angle 1:", font=font_small).grid(row=1, column=0, padx=2, pady=2, sticky="w")
for i in range(3):
    lbl = ttk.Label(engine_metrics_frame, text="0°", font=font_small)
    lbl.grid(row=1, column=i+1, padx=2, pady=2)
    servo_angle1_labels.append(lbl)

# Servo Angle 2
ttk.Label(engine_metrics_frame, text="Servo Angle 2:", font=font_small).grid(row=2, column=0, padx=2, pady=2, sticky="w")
for i in range(3):
    lbl = ttk.Label(engine_metrics_frame, text="0°", font=font_small)
    lbl.grid(row=2, column=i+1, padx=2, pady=2)
    servo_angle2_labels.append(lbl)

# Throttle %
ttk.Label(engine_metrics_frame, text="Throttle %:", font=font_small).grid(row=3, column=0, padx=2, pady=2, sticky="w")
for i in range(3):
    lbl = ttk.Label(engine_metrics_frame, text="0%", font=font_small)
    lbl.grid(row=3, column=i+1, padx=2, pady=2)
    throttle_labels.append(lbl)

# Thrust (N)
ttk.Label(engine_metrics_frame, text="Thrust (N):", font=font_small).grid(row=4, column=0, padx=2, pady=2, sticky="w")
for i in range(3):
    lbl = ttk.Label(engine_metrics_frame, text="0 N", font=font_small)
    lbl.grid(row=4, column=i+1, padx=2, pady=2)
    thrust_labels.append(lbl)

# --- 3D Plotting Setup (Each plot in its own LabelFrame) ---
# Main frame to hold both plot frames side-by-side
plots_main_frame = ttk.Frame(root, style='TFrame')
plots_main_frame.grid(row=2, column=1, padx=10, pady=5, sticky="nsew") # Placed in right column, no rowspan
plots_main_frame.grid_columnconfigure(0, weight=1)
plots_main_frame.grid_rowconfigure(0, weight=1)
plots_main_frame.grid_rowconfigure(1, weight=1)

# --- Raw Serial Data Display Section ---
raw_data_frame = ttk.LabelFrame(root, text="Raw Serial Data", style='TLabelframe')
raw_data_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
raw_data_frame.grid_rowconfigure(0, weight=1)
raw_data_frame.grid_columnconfigure(0, weight=1)

raw_data_text = tk.Text(
    raw_data_frame,
    wrap="none",
    height=10,
    bg=DARK_BG_COLOR,
    fg=LIGHT_TEXT_COLOR,
    insertbackground=LIGHT_TEXT_COLOR,
    relief="flat",
    borderwidth=0,
    highlightthickness=0
)
raw_data_text.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

raw_data_scrollbar_y = ttk.Scrollbar(raw_data_frame, orient="vertical", command=raw_data_text.yview)
raw_data_scrollbar_y.grid(row=0, column=1, sticky="ns")
raw_data_text.config(yscrollcommand=raw_data_scrollbar_y.set)

raw_data_scrollbar_x = ttk.Scrollbar(raw_data_frame, orient="horizontal", command=raw_data_text.xview)
raw_data_scrollbar_x.grid(row=1, column=0, sticky="ew")
raw_data_text.config(xscrollcommand=raw_data_scrollbar_x.set)

# Define plot background color to match the dark theme
PLOT_BG_COLOR = DARK_BG_COLOR # Now truly dark

# Attitude Plot Frame
attitude_plot_frame = ttk.LabelFrame(plots_main_frame, text="Attitude Visualization", style='Flat.TLabelframe')
attitude_plot_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew") # Grid within plots_main_frame
attitude_plot_frame.grid_rowconfigure(0, weight=1)
attitude_plot_frame.grid_columnconfigure(0, weight=1)

# Create a matplotlib figure for the Attitude plot (slightly smaller figsize)
fig_attitude = plt.figure(figsize=(1.8, 1.8))
# Set figure background color to match the LabelFrame
fig_attitude.patch.set_facecolor(PLOT_BG_COLOR)
ax_attitude = fig_attitude.add_subplot(111, projection='3d')
# Set axes background color to match the LabelFrame
ax_attitude.set_facecolor(PLOT_BG_COLOR)
canvas_attitude = FigureCanvasTkAgg(fig_attitude, master=attitude_plot_frame)
# Apply highlightthickness to the Tkinter widget returned by get_tk_widget()
canvas_attitude_tk_widget = canvas_attitude.get_tk_widget()
canvas_attitude_tk_widget.config(highlightthickness=0) # Set it here
canvas_attitude_tk_widget.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)


# Remove axis labels and titles
ax_attitude.set_xlabel("")
ax_attitude.set_ylabel("")
ax_attitude.set_zlabel("")
ax_attitude.set_title("")
ax_attitude.set_xticklabels([]) # Remove tick labels
ax_attitude.set_yticklabels([])
ax_attitude.set_zticklabels([])
ax_attitude.tick_params(axis='both', which='major', length=0) # Remove tick marks
ax_attitude.set_xlim([-1, 1])
ax_attitude.set_ylim([-1, 1])
ax_attitude.set_zlim([-1, 1])
ax_attitude.set_aspect('equal')
ax_attitude.grid(False) # Remove grid

initial_vectors = {
    'x': np.array([1, 0, 0]),
    'y': np.array([0, 1, 0]),
    'z': np.array([0, 0, 1])
}

quiver_x = ax_attitude.quiver(0, 0, 0, 0, 0, 0, color='r', length=0.8, arrow_length_ratio=0.1)
quiver_y = ax_attitude.quiver(0, 0, 0, 0, 0, 0, color='g', length=0.8, arrow_length_ratio=0.1)
quiver_z = ax_attitude.quiver(0, 0, 0, 0, 0, 0, color='b', length=0.8, arrow_length_ratio=0.1)


# Position Plot Frame
position_plot_frame = ttk.LabelFrame(plots_main_frame, text="Position Trail", style='Flat.TLabelframe')
position_plot_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew") # Grid within plots_main_frame
position_plot_frame.grid_rowconfigure(0, weight=1)
position_plot_frame.grid_columnconfigure(0, weight=1)

# Create a matplotlib figure for the Position plot (slightly smaller figsize)
fig_position = plt.figure(figsize=(2.0, 2.0))
# Set figure background color to match the LabelFrame
fig_position.patch.set_facecolor(PLOT_BG_COLOR)
ax_position = fig_position.add_subplot(111, projection='3d')
# Set axes background color to match the LabelFrame
ax_position.set_facecolor(PLOT_BG_COLOR)
canvas_position = FigureCanvasTkAgg(fig_position, master=position_plot_frame)
# Apply highlightthickness to the Tkinter widget returned by get_tk_widget()
canvas_position_tk_widget = canvas_position.get_tk_widget()
canvas_position_tk_widget.config(highlightthickness=0) # Set it here
canvas_position_tk_widget.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

# Remove axis labels and titles
ax_position.set_xlabel("")
ax_position.set_ylabel("")
ax_position.set_zlabel("")
ax_position.set_title("")
ax_position.set_xticklabels([]) # Remove tick labels
ax_position.set_yticklabels([])
ax_position.set_zticklabels([])
ax_position.tick_params(axis='both', which='major', length=0) # Remove tick marks
ax_position.set_xlim([-10, 10])
ax_position.set_ylim([-10, 10])
ax_position.set_zlim([-10, 10])
ax_position.set_aspect('equal')
ax_position.grid(False) # Remove grid

line_position_trail, = ax_position.plot([], [], [], 'w-', lw=1) # White line for trail

# Adjust subplot parameters for a tight layout for each figure
fig_attitude.tight_layout()
fig_position.tight_layout()


#
# --- Control Buttons Frame ---
control_frame = ttk.LabelFrame(root, text="Controls", style='TLabelframe')
control_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew", columnspan=2) # Now in row 3
control_frame.grid_columnconfigure(0, weight=1)
control_frame.grid_columnconfigure(1, weight=1)

# --- Predefined Command Buttons ---
# Function to populate the entry field with a predefined command
def populate_command_entry(command):
    """
    Sets the command in the entry field without triggering a send.
    """
    command_entry.delete(0, tk.END) # Clear existing text
    command_entry.insert(0, command) # Insert the predefined command


#
# Flight Commands
flight_commands_frame = ttk.LabelFrame(control_frame, text="Flight Commands", style='TLabelframe')
flight_commands_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
flight_commands_frame.grid_columnconfigure(0, weight=1)
flight_commands_frame.grid_columnconfigure(1, weight=1)

btn_auto_launch = ttk.Button(flight_commands_frame, text="Auto-Launch", command=lambda: populate_command_entry("auto-launch"), width=18)
btn_auto_launch.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
btn_ground_idle = ttk.Button(flight_commands_frame, text="Ground-Idle", command=lambda: populate_command_entry("ground-idle"), width=18)
btn_ground_idle.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
btn_abort_rtls = ttk.Button(flight_commands_frame, text="Abort-RTLS", command=lambda: populate_command_entry("Abort-RTLS"), width=18)
btn_abort_rtls.grid(row=1, column=0, padx=2, pady=2, sticky="ew")
btn_abort_direct = ttk.Button(flight_commands_frame, text="Abort-DIRECT", command=lambda: populate_command_entry("Abort-DIRECT"), width=18)
btn_abort_direct.grid(row=1, column=1, padx=2, pady=2, sticky="ew")
btn_freeze = ttk.Button(flight_commands_frame, text="Freeze", command=lambda: populate_command_entry("Freeze"), width=18)
btn_freeze.grid(row=2, column=0, padx=2, pady=2, sticky="ew", columnspan=2)


#
# Fan Control
fan_control_frame = ttk.LabelFrame(control_frame, text="Fan Control", style='TLabelframe')
fan_control_frame.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
fan_control_frame.grid_columnconfigure(0, weight=1)
fan_control_frame.grid_columnconfigure(1, weight=1)

btn_fan_10_idle = ttk.Button(fan_control_frame, text="Fan 10% Idle", command=lambda: populate_command_entry("fan 10% idle"), width=18)
btn_fan_10_idle.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
btn_fan_off = ttk.Button(fan_control_frame, text="Fan Off", command=lambda: populate_command_entry("fan off"), width=18)
btn_fan_off.grid(row=0, column=1, padx=2, pady=2, sticky="ew")


# List of all predefined command buttons for easy state management
predefined_command_buttons = [
    btn_auto_launch, btn_ground_idle, btn_abort_rtls, btn_abort_direct, btn_freeze,
    btn_fan_10_idle, btn_fan_off
]

# New: Entry field for typing commands (moved to bottom)
command_entry = ttk.Entry(control_frame, width=30)
command_entry.grid(row=2, column=0, padx=5, pady=5, sticky="ew", columnspan=1)

# New: Send Command button that uses the text from the entry field (moved to bottom)
def send_typed_command():
    global ser, running, start_time
    if not (ser and ser.is_open):
        messagebox.showwarning("Not Connected", "Please click 'Connect' in Serial Configuration first.")
        return

    if ser and ser.is_open:
        command_text = command_entry.get()
        if command_text:
            try:
                # Send the command followed by a newline character
                ser.write((command_text + '\n').encode('utf-8'))
                print(f"[DEBUG] Sent command: {command_text}")
                command_entry.delete(0, tk.END) # Clear the entry field after sending
            except Exception as e:
                messagebox.showerror("Send Error", f"Failed to send command: {e}")
                traceback.print_exc()
        else:
            messagebox.showwarning("No Command", "Please type a command to send.")
    else:
        messagebox.showerror("Not Connected", "Serial port is not open.")

send_typed_command_button = ttk.Button(control_frame, text="Send Typed Command", command=send_typed_command, state="disabled") # Disabled by default
send_typed_command_button.grid(row=2, column=1, padx=5, pady=5, sticky="ew")


# --- Functions ---
def list_serial_ports():
    """Lists available serial ports."""
    ports = serial.tools.list_ports.comports()
    return [p.device for p in ports]

def refresh_ports():
    """Refreshes the list of serial ports in the combobox."""
    port_combo["values"] = list_serial_ports()

def read_serial():
    """
    Reads data from the serial port in a separate thread.
    Parses quaternion and position data and stores them.
    Expected format: "DATA:qW,qX,qY,qZ,x,y,z"
    """
    global ser, running, start_time, engine_states, raw_serial_lines
    # Ensure start_time is set only once when the thread truly starts reading
    if start_time is None:
        start_time = time.time()
    while running:
        try:
            line = ser.readline() # Read a line from the serial port
            if not line:
                continue # Skip if no data
            decoded_line = line.decode("utf-8", errors="ignore").strip() # Decode and clean the line
            raw_serial_lines.append(decoded_line) # Store the raw line

            # Check if the line starts with "DATA:"
            if decoded_line.startswith("DATA:"):
                try:
                    parts = decoded_line.split(":")
                    data_str = parts[1].split(",") # Split all data components by comma
                    
                    # Parse quaternion components
                    qW = float(data_str[0])
                    qX = float(data_str[1])
                    qY = float(data_str[2])
                    qZ = float(data_str[3])

                    # SciPy's Rotation.from_quat() expects [x, y, z, w] order.
                    # We convert from [w, x, y, z] to [x, y, z, w].
                    latest_quaternion.append(np.array([qX, qY, qZ, qW]))

                    # Parse position components
                    x_pos = float(data_str[4])
                    y_pos = float(data_str[5])
                    z_pos = float(data_str[6])
                    
                    x_positions.append(x_pos)
                    y_positions.append(y_pos)
                    z_positions.append(z_pos)

                    # --- Placeholder for Engine State Update (Example) ---
                    # In a real scenario, you'd parse engine states from serial data.
                    # For demonstration, let's toggle engine 0 every 5 seconds of flight time.
                    # This will now toggle the first of the three engines.
                    if int(time.time() - start_time) % 5 == 0 and int(time.time() - start_time) != 0:
                        if not hasattr(read_serial, 'last_toggle_time'):
                            read_serial.last_toggle_time = 0
                        if time.time() - read_serial.last_toggle_time > 5:
                            engine_states[0] = not engine_states[0]
                            read_serial.last_toggle_time = time.time()


                except (ValueError, IndexError):
                    print(f"[ERROR] Could not parse data from line: {decoded_line}. Expected DATA:qW,qX,qY,qZ,x,y,z")
            else:
                # Optionally handle other types of serial data or log unparsed lines
                pass
        except Exception:
            print("Serial read error:")
            traceback.print_exc() # Print full traceback for debugging

def update_plot():
    """
    Updates the 3D attitude plot, 3D position trail plot, and telemetry data display.
    This function is called periodically by the Tkinter event loop.
    """
    global quiver_x, quiver_y, quiver_z, line_position_trail

    # Update current date and time
    current_time = time.strftime("%H:%M:%S")
    current_date = time.strftime("%Y-%m-%d")
    date_label.config(text=f"Date: {current_date}")
    time_label.config(text=f"Time: {current_time}")

    # Update Flight Time
    if start_time is not None and running:
        elapsed_time_s = time.time() - start_time
        hours, remainder = divmod(elapsed_time_s, 3600)
        minutes, seconds = divmod(remainder, 60)
        flight_time_value_label.config(text=f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}")
    else:
        flight_time_value_label.config(text="00:00:00")


    # --- Update Attitude Plot ---
    if latest_quaternion:
        q_scipy_format = latest_quaternion[0]
        try:
            rot = R.from_quat(q_scipy_format)
            rot_matrix = rot.as_matrix()

            transformed_x = rot_matrix @ initial_vectors['x']
            transformed_y = rot_matrix @ initial_vectors['y']
            transformed_z = rot_matrix @ initial_vectors['z']

            if quiver_x:
                quiver_x.remove()
                quiver_y.remove()
                quiver_z.remove()

            quiver_x = ax_attitude.quiver(0, 0, 0, transformed_x[0], transformed_x[1], transformed_x[2],
                                 color='r', length=0.8, arrow_length_ratio=0.1)
            quiver_y = ax_attitude.quiver(0, 0, 0, transformed_y[0], transformed_y[1], transformed_y[2],
                                 color='g', length=0.8, arrow_length_ratio=0.1)
            quiver_z = ax_attitude.quiver(0, 0, 0, transformed_z[0], transformed_z[1], transformed_z[2],
                                 color='b', length=0.8, arrow_length_ratio=0.1)
            
        except Exception as e:
            print(f"Error updating attitude plot: {e}")
            traceback.print_exc()

    # --- Update Position Plot ---
    if x_positions and y_positions and z_positions:
        try:
            line_position_trail.set_data(list(x_positions), list(y_positions))
            line_position_trail.set_3d_properties(list(z_positions))

            # Dynamically adjust limits for the position plot
            if len(x_positions) > 1:
                min_x, max_x = min(x_positions), max(x_positions)
                min_y, max_y = min(y_positions), max(y_positions)
                min_z, max_z = min(z_positions), max(z_positions)

                buffer = 1.0
                ax_position.set_xlim([min_x - buffer, max_x + buffer])
                ax_position.set_ylim([min_y - buffer, max_y + buffer])
                ax_position.set_zlim([min_z - buffer, max_z + buffer])
            else:
                ax_position.set_xlim([-10, 10])
                ax_position.set_ylim([-10, 10])
                ax_position.set_zlim([-10, 10])

        except Exception as e:
            print(f"Error updating position plot: {e}")
            traceback.print_exc()

    # --- Update Engine Visuals ---
    update_engine_visuals()

    # --- Update Raw Serial Data Display ---
    update_raw_serial_display()

    # Redraw both canvases
    canvas_attitude.draw_idle()
    canvas_position.draw_idle()

    # Schedule the next plot update after 50 milliseconds for smoother animation
    root.after(50, update_plot)

def update_engine_visuals():
    for i in range(3):
        color = ACCENT_GREEN if engine_states[i] else DARK_BG_COLOR
        engine_canvas.itemconfigure(f"eng{i}", fill=color)

def update_raw_serial_display():
    """Updates the raw serial data text widget with the latest lines."""
    # Only update if there's new data
    if raw_serial_lines:
        raw_data_text.config(state="normal") # Enable editing temporarily
        raw_data_text.delete("1.0", tk.END) # Clear existing text

        # Insert all lines from the deque
        for line in raw_serial_lines:
            raw_data_text.insert(tk.END, line + "\n")
        
        raw_data_text.see(tk.END) # Scroll to the bottom
        raw_data_text.config(state="disabled") # Disable editing

def set_command_control_states(state):
    """Helper function to set the state of command buttons and entry."""
    send_typed_command_button.config(state=state)
    command_entry.config(state=state)
    for btn in predefined_command_buttons:
        btn.config(state=state)

def connect_serial():
    """
    Establishes a serial connection based on user selection.
    """
    global ser, running, start_time
    port = port_combo.get()
    baud = baud_combo.get()

    if not port:
        messagebox.showerror("Error", "Please select a serial port.")
        return

    try:
        ser = serial.Serial(port, int(baud), timeout=0.1)
        running = True
        threading.Thread(target=read_serial, daemon=True).start()
        
        connect_button.config(state="disabled")
        disconnect_button.config(state="normal") # Enable disconnect button
        port_combo.config(state="disabled")
        baud_combo.config(state="disabled")
        set_command_control_states("normal") # Enable all command controls
        connection_status_label.config(text="● Connected", foreground=ACCENT_GREEN)
        start_time = time.time() # Set start time when connected
    except Exception as e:
        messagebox.showerror("Connection Failed", str(e))
        traceback.print_exc()

def disconnect_serial():
    """
    Closes the serial connection and resets GUI elements and plot data.
    """
    global ser, running, start_time
    if ser and ser.is_open:
        running = False
        ser.close()
        ser = None
        messagebox.showinfo("Disconnected", "Serial port disconnected.")
    
    # Reset GUI elements for a disconnected state
    connect_button.config(state="normal")
    disconnect_button.config(state="disabled") # Disable disconnect button
    port_combo.config(state="normal")
    baud_combo.config(state="normal")
    set_command_control_states("disabled") # Disable all command controls
    connection_status_label.config(text="● Disconnected", foreground=ACCENT_RED)
    
    # Reset flight time
    start_time = None
    flight_time_value_label.config(text="00:00:00")

    # Reset plots to default
    latest_quaternion.clear()
    latest_quaternion.append(np.array([0, 0, 0, 1])) # Reset attitude to no rotation
    
    x_positions.clear()
    y_positions.clear()
    z_positions.clear()
    
    # Reset position plot line
    line_position_trail.set_data([], [])
    line_position_trail.set_3d_properties([])
    ax_position.set_xlim([-10, 10]) # Reset position plot limits
    ax_position.set_ylim([-10, 10])
    ax_position.set_zlim([-10, 10])

    # Reset engine states
    for i in range(len(engine_states)):
        engine_states[i] = False
    update_engine_visuals() # Immediately update the display

    # Clear raw serial data display
    raw_serial_lines.clear()
    raw_data_text.config(state="normal")
    raw_data_text.delete("1.0", tk.END)
    raw_data_text.config(state="disabled")

    canvas_attitude.draw_idle() # Redraw to clear old vectors
    canvas_position.draw_idle() # Redraw to clear old trail

# --- Main Application Flow ---
refresh_ports() # Populate ports when the application starts
connect_button.config(command=connect_serial) # Assign command to connect button
disconnect_button.config(state="disabled") # Disable disconnect button initially
set_command_control_states("disabled") # Disable command controls initially

# Start the periodic update for the plot and data display
update_plot()

root.mainloop()