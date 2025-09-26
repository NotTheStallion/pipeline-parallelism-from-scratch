import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_pipeline_schedule_png(num_gpus, num_microbatches, time_per_op=0.3, filename="pipeline_schedule.png"):
    """
    Create a pipeline schedule visualization and save as PNG
    
    Args:
        num_gpus: Number of GPUs in the pipeline
        num_microbatches: Number of microbatches
        time_per_op: Time per operation (default 0.3)
        filename: Output PNG filename
    """
    
    # Calculate total time slots needed (simplified calculation)
    total_time_slots = num_gpus + num_microbatches * 2 - 2
    
    # Create figure with appropriate size
    fig_width = max(12, num_gpus * 2)
    fig_height = max(8, total_time_slots * 0.4)
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, fig_height)
    ax.invert_yaxis()  # So time goes from top to bottom
    ax.axis('off')  # Hide axes
    
    # Colors for different operations
    colors = {
        'forward': '#d5e8d4',  # Green
        'backward': '#f8cecc',  # Red
        'weight': '#fff2cc',   # Yellow
        'optimizer': '#e1d5e7', # Purple
        'idle': '#f5f5f5',     # Gray
        'header': '#dae8fc'    # Blue for headers
    }
    
    # Cell dimensions
    cell_width = 1.8
    cell_height = 0.35
    header_height = 0.5
    time_column_width = 1.5
    margin_x = 0.5
    margin_y = 0.5
    
    # Generate pipeline schedule data
    schedule_data = generate_detailed_pipeline_data(num_gpus, num_microbatches)
    
    # Draw headers
    y_pos = margin_y
    
    # Time header
    draw_rounded_rect(ax, margin_x, y_pos, time_column_width, header_height, 
                     colors['header'], "Time (s)")
    
    # GPU headers
    x_pos = margin_x + time_column_width
    for gpu in range(num_gpus):
        draw_rounded_rect(ax, x_pos, y_pos, cell_width, header_height, 
                         colors['header'], f"GPU {gpu+1}")
        x_pos += cell_width
    
    # Operation type header
    draw_rounded_rect(ax, x_pos, y_pos, cell_width * 1.5, header_height, 
                     colors['header'], "Operation Type")
    
    y_pos += header_height
    
    # Draw schedule rows
    for time_slot in range(total_time_slots):
        x_pos = margin_x
        
        # Time label
        time_value = time_slot * time_per_op
        draw_rounded_rect(ax, x_pos, y_pos, time_column_width, cell_height, 
                         '#f0f0f0', f"{time_value:.1f}", fontsize=9)
        x_pos += time_column_width
        
        # GPU operations
        current_operations = []
        for gpu in range(num_gpus):
            if time_slot < len(schedule_data[gpu]):
                operation = schedule_data[gpu][time_slot]
            else:
                operation = "idle"
            
            # Determine color and text
            if operation.startswith('F'):
                color = colors['forward']
                text = operation
                current_operations.append('Forward')
            elif operation.startswith('B'):
                color = colors['backward']
                text = operation
                current_operations.append('Backward')
            elif operation.startswith('W'):
                color = colors['weight']
                text = operation
                current_operations.append('Weight')
            elif operation.startswith('O'):
                color = colors['optimizer']
                text = operation
                current_operations.append('Optimizer')
            else:
                color = colors['idle']
                text = "idle"
                current_operations.append('Idle')
            
            draw_rounded_rect(ax, x_pos, y_pos, cell_width, cell_height, color, text, fontsize=8)
            x_pos += cell_width
        
        # Operation type description
        op_type = get_operation_description(current_operations)
        draw_rounded_rect(ax, x_pos, y_pos, cell_width * 1.5, cell_height, 
                         '#e1d5e7', op_type, fontsize=8)
        
        y_pos += cell_height
    
    # Add legend
    y_pos += 0.3
    x_pos = margin_x
    legend_items = [
        ("Forward Pass", colors['forward']),
        ("Backward Pass", colors['backward']),
        ("Weight Update", colors['weight']),
        ("Optimizer Step", colors['optimizer']),
        ("Device Idle", colors['idle'])
    ]
    
    for label, color in legend_items:
        draw_rounded_rect(ax, x_pos, y_pos, cell_width, cell_height * 0.8, color, label, fontsize=8)
        x_pos += cell_width + 0.1
    
    # Add title
    plt.suptitle(f'Pipeline Parallel Schedule: {num_gpus} GPUs, {num_microbatches} Microbatches', 
                 fontsize=14, fontweight='bold', y=0.95)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Pipeline schedule saved as: {filename}")
    print(f"Configuration: {num_gpus} GPUs, {num_microbatches} microbatches")
    print(f"Total simulated time: {total_time_slots * time_per_op:.1f} seconds")

def draw_rounded_rect(ax, x, y, width, height, color, text, fontsize=10):
    """Draw a rounded rectangle with text"""
    rect = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor='black', linewidth=0.5)
    ax.add_patch(rect)
    
    # Add text
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', fontsize=fontsize, weight='bold')

def generate_detailed_pipeline_data(num_gpus, num_microbatches):
    """
    Generate more realistic pipeline schedule data
    """
    schedule = [[] for _ in range(num_gpus)]
    total_time_slots = num_gpus + num_microbatches * 2 - 2
    
    # Fill phase: Forward passes flowing through the pipeline
    for time in range(total_time_slots):
        for gpu in range(num_gpus):
            if time < len(schedule[gpu]):
                continue  # Already filled
            
            # Fill phase logic
            if time >= gpu and time - gpu < num_microbatches:
                microbatch = time - gpu + 1
                schedule[gpu].append(f"F{microbatch}")
            # Steady state and flush phase logic
            elif time >= num_gpus + gpu and time - (num_gpus + gpu) < num_microbatches:
                microbatch = time - (num_gpus + gpu) + 1
                schedule[gpu].append(f"B{microbatch}")
            else:
                # Add idle slots or continue previous operations
                if time < len(schedule[gpu]):
                    continue
                else:
                    schedule[gpu].append("idle")
    
    # Ensure all GPU schedules have the same length
    max_len = max(len(sched) for sched in schedule)
    for sched in schedule:
        while len(sched) < max_len:
            sched.append("idle")
    
    return schedule

def get_operation_description(operations):
    """Get description of operations happening in a time slot"""
    op_counts = {}
    for op in operations:
        op_type = op.split('(')[0] if '(' in op else op
        op_counts[op_type] = op_counts.get(op_type, 0) + 1
    
    if len(op_counts) == 1:
        op_type = list(op_counts.keys())[0]
        if op_type == 'Forward':
            return "Forward Pass"
        elif op_type == 'Backward':
            return "Backward Pass"
        elif op_type == 'Idle':
            return "Pipeline Fill/Flush"
        else:
            return op_type
    
    # Mixed operations
    if 'Forward' in op_counts and 'Backward' in op_counts:
        return "1F1B Steady State"
    elif 'Forward' in op_counts:
        return "Pipeline Filling"
    elif 'Backward' in op_counts:
        return "Pipeline Flushing"
    else:
        return "Mixed Operations"

def create_complex_pipeline_schedule(num_gpus, num_microbatches, filename="complex_pipeline.png"):
    """
    Create a more complex and realistic pipeline schedule
    """
    # More sophisticated scheduling algorithm
    schedule = [[] for _ in range(num_gpus)]
    total_time = num_gpus + num_microbatches + (num_microbatches - 1)
    
    # Phase 1: Pipeline filling (only forward passes)
    for t in range(total_time):
        for stage in range(num_gpus):
            if t < stage:
                # GPU not active yet
                if len(schedule[stage]) <= t:
                    schedule[stage].append("idle")
                continue
                
            microbatch = t - stage + 1
            if microbatch > num_microbatches:
                # No more forward passes for this stage
                if len(schedule[stage]) <= t:
                    schedule[stage].append("idle")
                continue
            
            if t - stage < num_microbatches:
                # Forward pass
                schedule[stage].append(f"F{microbatch}")
            else:
                # Should be backward pass or idle
                backward_microbatch = microbatch - num_gpus + 1
                if backward_microbatch > 0:
                    schedule[stage].append(f"B{backward_microbatch}")
                else:
                    schedule[stage].append("idle")
    
    # Ensure equal length
    max_len = max(len(s) for s in schedule)
    for s in schedule:
        while len(s) < max_len:
            s.append("idle")
    
    # Create the visualization
    create_visualization(schedule, num_gpus, num_microbatches, filename)

def create_visualization(schedule, num_gpus, num_microbatches, filename):
    """Create visualization from schedule data"""
    time_per_op = 0.3
    total_time_slots = len(schedule[0])
    
    fig, ax = plt.subplots(figsize=(14, max(6, total_time_slots * 0.3)))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, total_time_slots * 0.4 + 2)
    ax.invert_yaxis()
    ax.axis('off')
    
    colors = {
        'F': '#d5e8d4',  # Forward - Green
        'B': '#f8cecc',  # Backward - Red
        'W': '#fff2cc',  # Weight - Yellow
        'O': '#e1d5e7',  # Optimizer - Purple
        'idle': '#f5f5f5'
    }
    
    # Draw headers
    y = 0.5
    cell_width = 1.2
    cell_height = 0.3
    
    # Time header
    ax.add_patch(plt.Rectangle((0.5, y), 1.0, 0.5, facecolor='#dae8fc', edgecolor='black'))
    ax.text(1.0, y + 0.25, 'Time (s)', ha='center', va='center', weight='bold')
    
    # GPU headers
    x = 1.5
    for i in range(num_gpus):
        ax.add_patch(plt.Rectangle((x, y), cell_width, 0.5, facecolor='#dae8fc', edgecolor='black'))
        ax.text(x + cell_width/2, y + 0.25, f'GPU {i+1}', ha='center', va='center', weight='bold')
        x += cell_width
    
    y += 0.6
    
    # Draw schedule
    for t in range(total_time_slots):
        x = 0.5
        # Time label
        ax.add_patch(plt.Rectangle((x, y), 1.0, cell_height, facecolor='#f0f0f0', edgecolor='black'))
        ax.text(x + 0.5, y + cell_height/2, f'{t * time_per_op:.1f}', ha='center', va='center', fontsize=8)
        x += 1.0
        
        # GPU operations
        for gpu in range(num_gpus):
            op = schedule[gpu][t] if t < len(schedule[gpu]) else "idle"
            color = colors.get(op[0] if op != "idle" else "idle", colors['idle'])
            
            ax.add_patch(plt.Rectangle((x, y), cell_width, cell_height, facecolor=color, edgecolor='black'))
            ax.text(x + cell_width/2, y + cell_height/2, op, ha='center', va='center', fontsize=8)
            x += cell_width
        
        y += cell_height + 0.05
    
    # Add title and legend
    plt.title(f'Pipeline Parallel Training: {num_gpus} GPUs, {num_microbatches} Microbatches', 
              pad=20, fontsize=14, weight='bold')
    
    # Save
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")

# Example usage
if __name__ == "__main__":
    # Create schedule for 4 GPUs, 8 microbatches (as requested)
    create_pipeline_schedule_png(4, 8, filename="pipeline_4gpu_8mb.png")
    
    # Create schedule for original 3 GPUs, 6 microbatches
    create_pipeline_schedule_png(3, 6, filename="pipeline_3gpu_6mb.png")
    
    # Create more complex visualization
    create_complex_pipeline_schedule(4, 8, filename="complex_pipeline_4gpu_8mb.png")
    
    # Additional examples
    create_pipeline_schedule_png(2, 4, filename="pipeline_2gpu_4mb.png")
    create_pipeline_schedule_png(8, 16, filename="pipeline_8gpu_16mb.png")