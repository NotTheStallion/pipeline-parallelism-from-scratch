import xml.etree.ElementTree as ET
from xml.dom import minidom

def generate_pipeline_schedule(num_gpus, num_microbatches, time_per_op=0.3):
    """
    Generate a pipeline parallel schedule for Draw.io
    
    Args:
        num_gpus: Number of GPUs in the pipeline
        num_microbatches: Number of microbatches
        time_per_op: Time per operation (default 0.3)
    """
    
    # Calculate total time slots needed
    total_time_slots = num_gpus + num_microbatches - 1 + (num_microbatches - 1)
    
    # Create XML structure
    mxfile = ET.Element('mxfile', {
        'host': 'app.diagrams.net',
        'modified': '2024-01-01T00:00:00.000Z',
        'agent': '5.0 (Python Script)',
        'version': '22.1.6',
        'type': 'device'
    })
    
    diagram = ET.SubElement(mxfile, 'diagram', {
        'id': 'PipelineSchedule',
        'name': f'Pipeline Schedule - {num_gpus} GPUs, {num_microbatches} Microbatches'
    })
    
    mxGraphModel = ET.SubElement(diagram, 'mxGraphModel', {
        'dx': '1426',
        'dy': '914',
        'grid': '1',
        'gridSize': '10',
        'guides': '1',
        'tooltips': '1',
        'connect': '1',
        'arrows': '1',
        'fold': '1',
        'page': '1',
        'pageScale': '1',
        'pageWidth': '827',
        'pageHeight': '1169',
        'math': '0',
        'shadow': '0'
    })
    
    root = ET.SubElement(mxGraphModel, 'root')
    
    # Add background cell
    ET.SubElement(root, 'mxCell', {'id': '0'})
    ET.SubElement(root, 'mxCell', {'id': '1', 'parent': '0'})
    
    # Calculate positions and dimensions
    cell_width = 80
    cell_height = 40
    header_height = 60
    time_column_width = 100
    margin = 50
    
    # Create time labels
    time_labels = []
    for i in range(total_time_slots + 1):
        time_labels.append(f"{(i) * time_per_op:.1f}")
    
    # Create headers
    headers = ["Time"] + [f"GPU {i+1}" for i in range(num_gpus)] + ["Operation Type"]
    
    # Create grid cells
    y_pos = margin
    
    # Create header row
    x_pos = margin
    for i, header in enumerate(headers):
        width = time_column_width if i == 0 else cell_width
        cell = ET.SubElement(root, 'mxCell', {
            'id': f'header_{i}',
            'value': header,
            'style': 'swimlane;fontStyle=1;align=center;verticalAlign=middle;fillColor=#dae8fc;strokeColor=#6c8ebf;',
            'parent': '1',
            'vertex': '1'
        })
        geometry = ET.SubElement(cell, 'mxGeometry', {
            'x': str(x_pos),
            'y': str(y_pos),
            'width': str(width),
            'height': str(header_height)
        })
        x_pos += width
    
    y_pos += header_height
    
    # Generate pipeline schedule data
    schedule_data = generate_pipeline_data(num_gpus, num_microbatches)
    
    # Create data rows
    for time_slot in range(total_time_slots):
        x_pos = margin
        
        # Time label
        cell = ET.SubElement(root, 'mxCell', {
            'id': f'time_{time_slot}',
            'value': time_labels[time_slot],
            'style': 'align=center;verticalAlign=middle;fillColor=#f5f5f5;strokeColor=#666666;',
            'parent': '1',
            'vertex': '1'
        })
        ET.SubElement(cell, 'mxGeometry', {
            'x': str(x_pos),
            'y': str(y_pos),
            'width': str(time_column_width),
            'height': str(cell_height)
        })
        x_pos += time_column_width
        
        # GPU cells
        for gpu in range(num_gpus):
            operation = schedule_data[gpu][time_slot] if time_slot < len(schedule_data[gpu]) else "idle"
            
            # Determine cell color based on operation type
            if "F" in operation:
                fill_color = "#d5e8d4"  # Green for forward
            elif "B" in operation:
                fill_color = "#f8cecc"  # Red for backward
            elif "W" in operation or "O" in operation:
                fill_color = "#fff2cc"  # Yellow for weight/optimizer
            else:
                fill_color = "#f5f5f5"  # Gray for idle
            
            cell = ET.SubElement(root, 'mxCell', {
                'id': f'gpu{gpu}_time{time_slot}',
                'value': operation,
                'style': f'align=center;verticalAlign=middle;fillColor={fill_color};strokeColor=#666666;',
                'parent': '1',
                'vertex': '1'
            })
            ET.SubElement(cell, 'mxGeometry', {
                'x': str(x_pos),
                'y': str(y_pos),
                'width': str(cell_width),
                'height': str(cell_height)
            })
            x_pos += cell_width
        
        # Operation type description
        op_type = get_operation_type(schedule_data, time_slot, num_gpus)
        cell = ET.SubElement(root, 'mxCell', {
            'id': f'op_type_{time_slot}',
            'value': op_type,
            'style': 'align=center;verticalAlign=middle;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=10;',
            'parent': '1',
            'vertex': '1'
        })
        ET.SubElement(cell, 'mxGeometry', {
            'x': str(x_pos),
            'y': str(y_pos),
            'width': str(cell_width * 2),
            'height': str(cell_height)
        })
        
        y_pos += cell_height
    
    # Add legend
    y_pos += 20
    legend_items = [
        ("Forward Pass", "#d5e8d4"),
        ("Backward Pass", "#f8cecc"),
        ("Weight Update", "#fff2cc"),
        ("Idle", "#f5f5f5")
    ]
    
    x_pos = margin
    for label, color in legend_items:
        cell = ET.SubElement(root, 'mxCell', {
            'id': f'legend_{label.replace(" ", "_")}',
            'value': label,
            'style': f'align=center;verticalAlign=middle;fillColor={color};strokeColor=#666666;',
            'parent': '1',
            'vertex': '1'
        })
        ET.SubElement(cell, 'mxGeometry', {
            'x': str(x_pos),
            'y': str(y_pos),
            'width': str(cell_width),
            'height': str(30)
        })
        x_pos += cell_width + 10
    
    return prettify(mxfile)

def generate_pipeline_data(num_gpus, num_microbatches):
    """
    Generate pipeline schedule data for the given configuration
    """
    schedule = [[] for _ in range(num_gpus)]
    total_time_slots = num_gpus + num_microbatches - 1 + (num_microbatches - 1)
    
    # Fill phase
    for stage in range(num_gpus):
        for time in range(stage + 1):
            if time < num_microbatches:
                microbatch = time + 1
                schedule[stage].append(f"F{microbatch}")
    
    # Steady state and flush phase
    for time in range(num_gpus, total_time_slots):
        for stage in range(num_gpus):
            if len(schedule[stage]) <= time:
                # Determine if it's forward or backward
                if stage == 0:  # First stage
                    microbatch = time - stage + 1
                    if microbatch <= num_microbatches:
                        schedule[stage].append(f"F{microbatch}")
                    else:
                        schedule[stage].append("idle")
                elif stage == num_gpus - 1:  # Last stage
                    microbatch = time - stage - (num_gpus - 2) + 1
                    if 1 <= microbatch <= num_microbatches:
                        schedule[stage].append(f"B{microbatch}")
                    else:
                        schedule[stage].append("idle")
                else:  # Middle stages
                    # Simplified logic - in real implementation this would be more complex
                    microbatch_f = time - stage + 1
                    microbatch_b = time - stage - (num_gpus - 2) + 1
                    
                    if microbatch_f <= num_microbatches:
                        schedule[stage].append(f"F{microbatch_f}")
                    elif microbatch_b >= 1:
                        schedule[stage].append(f"B{microbatch_b}")
                    else:
                        schedule[stage].append("idle")
    
    # Ensure all GPU schedules have the same length
    max_len = max(len(sched) for sched in schedule)
    for sched in schedule:
        while len(sched) < max_len:
            sched.append("idle")
    
    return schedule

def get_operation_type(schedule_data, time_slot, num_gpus):
    """Determine the operation type for a given time slot"""
    operations = []
    for gpu in range(num_gpus):
        if time_slot < len(schedule_data[gpu]):
            op = schedule_data[gpu][time_slot]
            if "F" in op:
                operations.append("Forward")
            elif "B" in op:
                operations.append("Backward")
    
    if not operations:
        return "Device idle"
    
    # Count unique operation types
    unique_ops = set(operations)
    if len(unique_ops) == 1:
        return f"{operations[0]} Pass"
    else:
        return "Pipeline Steady State"

def prettify(elem):
    """Return a pretty-printed XML string for the Element"""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def save_drawio_file(filename, num_gpus, num_microbatches):
    """Generate and save a Draw.io file"""
    xml_content = generate_pipeline_schedule(num_gpus, num_microbatches)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(xml_content)
    
    print(f"Draw.io file saved as: {filename}")

# Example usage
if __name__ == "__main__":
    # Generate for 4 GPUs, 8 microbatches (as requested)
    save_drawio_file("pipeline_4gpu_8mb.drawio", 4, 8)
    
    # Generate for the original 3 GPUs, 6 microbatches
    save_drawio_file("pipeline_3gpu_6mb.drawio", 3, 6)
    
    # You can generate for any configuration
    save_drawio_file("pipeline_2gpu_4mb.drawio", 2, 4)