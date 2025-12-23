import os
import networkx as nx
from pyvis.network import Network
from flask import Flask, render_template_string, request, send_file
import glob

app = Flask(__name__)

BASE_DIR = '/data/zxl/Search2026/outputData'

def find_gml_files():
    gml_files = []
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith('.gml'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, BASE_DIR)
                gml_files.append(rel_path)
    return sorted(gml_files)

@app.route('/')
def index():
    files = find_gml_files()
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GML Visualizer</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            ul { list-style-type: none; padding: 0; }
            li { margin: 5px 0; }
            a { text-decoration: none; color: #007bff; }
            a:hover { text-decoration: underline; }
            .container { max-width: 800px; margin: 0 auto; }
            h1 { color: #333; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Select a GML file to visualize</h1>
            <ul>
                {% for file in files %}
                <li><a href="/view/{{ file }}">{{ file }}</a></li>
                {% endfor %}
            </ul>
        </div>
    </body>
    </html>
    """
    return render_template_string(html, files=files)

@app.route('/view/<path:filename>')
def view_graph(filename):
    file_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(file_path):
        return "File not found", 404

    try:
        G = nx.read_gml(file_path)
    except Exception as e:
        return f"Error reading GML file: {e}", 500

    # Create Pyvis network
    # Using CDN for resources to avoid local dependency issues and ensure TomSelect is available
    # Changed background to white and font color to black as requested
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black", select_menu=True, filter_menu=True, cdn_resources='remote')
    
    # Process nodes for visualization
    for node_id in G.nodes():
        node = G.nodes[node_id]
        
        # Set label to sig if available, else use existing label or id
        if 'sig' in node:
            sig_str = str(node['sig'])
            # Simplify sig: split by '.' and keep last two parts
            parts = sig_str.split('.')
            if len(parts) >= 2:
                node['label'] = f"{parts[-2]}.{parts[-1]}"
            else:
                node['label'] = sig_str
        
        # Construct title (tooltip) with all attributes
        title_parts = []
        for key, value in node.items():
            # Format long values (like code) to be more readable if needed
            val_str = str(value)
            if len(val_str) > 100:
                val_str = val_str[:100] + "..."
            title_parts.append(f"<b>{key}:</b> {val_str}")
        
        node['title'] = "<br>".join(title_parts)
        
        # Update node in graph
        G.nodes[node_id].update(node)

    # Process edges for visualization
    for u, v, data in G.edges(data=True):
        # Add edge type label if available (assuming 'type' or similar attribute exists, or just show all attributes in title)
        # If you specifically want to show a label on the edge, pyvis uses 'label' attribute
        # Let's check if there's a specific attribute for edge type. 
        # If not specified, we might just look for common ones or just put everything in title.
        # But user asked to "show edge type on edge". Let's assume there is an attribute that represents type.
        # Common GML edge attributes might be 'label', 'relation', 'type', etc.
        # Without seeing edge data, I will try to find a likely candidate or just iterate all to find something meaningful.
        # For now, let's put all data in title, and if there is a 'type' or 'relation' or 'label', use it as label.
        
        edge_title_parts = []
        label_candidate = None
        
        for key, value in data.items():
             edge_title_parts.append(f"<b>{key}:</b> {value}")
             if key.lower() in ['type', 'relation', 'rel', 'label']:
                 label_candidate = str(value)
        
        data['title'] = "<br>".join(edge_title_parts)
        if label_candidate:
            data['label'] = label_candidate
            
        G.edges[u, v].update(data)

    net.from_nx(G)
    
    # Set options for better physics/layout
    net.set_options("""
    var options = {
      "nodes": {
        "font": {
          "size": 12
        },
        "shape": "dot",
        "size": 10
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {
          "enabled": true,
          "iterations": 1000
        }
      }
    }
    """)
    
    # Save to a temporary file and serve it
    output_path = os.path.join(os.path.dirname(file_path), "temp_viz.html")
    # Make sure we don't overwrite anything important, maybe use a temp dir but for now next to file is okay or just render string
    # Pyvis write_html writes to file. generate_html returns string.
    
    html_content = net.generate_html()
    return html_content

if __name__ == '__main__':
    # Use 0.0.0.0 to be accessible if needed, though usually localhost is fine in this env
    app.run(host='0.0.0.0', port=5000, debug=True)
