import os
import gradio as gr
import subprocess
import json

# Configuration paths (you might want to adjust these)
CONFIG = {
    'model_path': "OpenGVLab/InternVL2-8B",
    'get_boundary_model_path': "Yongxin-Guo/trace-uni",
    'get_mask_model_path': "./checkpoints/sam2.1_hiera_base_plus.pt",
    'output_folder': "./results/",
    'frame_count': 16
}

def run_inference_pipeline(video_path, object_bbox_path=None):
    """
    Run the entire inference pipeline for video processing
    """
    # Ensure output folder exists
    os.makedirs(CONFIG['output_folder'], exist_ok=True)

    # Prepare file paths
    video_name = os.path.basename(video_path)
    qa_file_path = os.path.join(CONFIG['output_folder'], f"{os.path.splitext(video_name)[0]}_boundary.json")
    final_json_path = os.path.join(CONFIG['output_folder'], f"{os.path.splitext(video_name)[0]}_boundary_caption.json")
    final_video_path = os.path.join(CONFIG['output_folder'], f"{os.path.splitext(video_name)[0]}_boundary_caption.mp4")
    masked_video_path = os.path.join(CONFIG['output_folder'], f"{os.path.splitext(video_name)[0]}_mask.mp4")

    # Prepare commands
    commands = [
        # Step 1: Parsing/Boundary Detection
        f"python -m scripts.get_boundary "
        f"--video_paths {video_path} "
        f"--questions 'Localize a series of activity events in the video, output the start and end timestamp for each event, and describe each event with sentences.' "
        f"--model_path {CONFIG['get_boundary_model_path']}",

        # Optional: Segmentation (only if bbox is provided)
        (f"python scripts/get_masks.py "
         f"--video_path {video_path} "
         f"--txt_path {object_bbox_path} "
         f"--model_path {CONFIG['get_mask_model_path']} "
         f"--video_output_path {CONFIG['output_folder']} "
         f"--save_to_video True") if object_bbox_path else None,

        # Step 2: Captioning
        f"python scripts/get_caption.py "
        f"--model_path {CONFIG['model_path']} "
        f"--QA_file_path {qa_file_path} "
        f"--video_folder {CONFIG['output_folder']} "
        f"--answers_output_folder {CONFIG['output_folder']} "
        f"--extract_frames_method max_frames_num "
        f"--max_frames_num {CONFIG['frame_count']} "
        f"--frames_from video "
        f"--provide_boundaries",

        # Step 3: Generate Visualization
        f"python scripts/get_vis.py {masked_video_path} {final_json_path} {final_video_path}"
    ]

    # Filter out None commands
    commands = [cmd for cmd in commands if cmd is not None]

    # Execute commands
    for cmd in commands:
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error in command: {cmd}")
            print(f"Error details: {e}")
            return None

    # Read and return results
    try:
        with open(final_json_path, 'r') as f:
            results = json.load(f)
        return {
            'captions': results,
            'final_video': final_video_path
        }
    except Exception as e:
        print(f"Error reading results: {e}")
        return None

def caption_video(video, bbox_file=None):
    """
    Gradio-friendly wrapper for inference pipeline
    """
    if video is None:
        return "Please upload a video first.", None

    results = run_inference_pipeline(video, bbox_file)
    
    if results is None:
        return "Processing failed. Please check the logs.", None
    
    # Format captions nicely
    captions_text = "\n\n".join([
        f"Event {i+1} (Time: {event.get('timestamp', 'N/A')}):\n{event.get('caption', 'No caption')}"
        for i, event in enumerate(results.get('captions', []))
    ])

    return captions_text, results.get('final_video')

def create_demo():
    """
    Create Gradio interface
    """

    DESCRIPTION = '''# CAT2: 
    This is a demo for our ''CAT2'' [paper](https://github.com/yunlong10/CAT-2). Code is available [here](https://github.com/yunlong10/CAT-2)
    This demo preform captioning ...
    '''

    with gr.Blocks() as demo:
        gr.Markdown("# Caption Anything Demo")
        gr.Markdown(DESCRIPTION)
        gr.Markdown("Upload a video and optionally a bounding box text file for detailed captioning.")
        
        with gr.Row():
            video_input = gr.Video(label="Upload Video")
            bbox_input = gr.File(label="Upload Bounding Box File (Optional)", type="file")
        
        caption_button = gr.Button("Generate Captions")
        
        output_text = gr.Textbox(label="Video Captions")
        output_video = gr.Video(label="Processed Video")
        
        caption_button.click(
            fn=caption_video, 
            inputs=[video_input, bbox_input],
            outputs=[output_text, output_video]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",  # Make accessible from other machines
        server_port=7860,
        debug=True
    )