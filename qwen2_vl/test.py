import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys_paths = [
    os.path.abspath(os.path.join(script_dir, '../'))
]
print(sys_paths)
for sys_path in sys_paths:
    if sys_path not in sys.path:
        sys.path.append(sys_path)
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from tqdm import tqdm
from eval_utils import gen_QAs, basic_parser, get_answers_output_path, gen_QAs_dataloader, unpack_QAs
import json

# default: Load the model on the available device(s)
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.

parser = basic_parser()
args = parser.parse_args()

model = Qwen2VLForConditionalGeneration.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained(args.model_path)

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

dataloader = gen_QAs_dataloader(args)
QAs = gen_QAs(args.QA_file_path, args.video_folder)
answers_output_path = get_answers_output_path(args)
answers = []
for idx, (QA_data,_) in enumerate(tqdm(dataloader)):
    try:  
        final_question, video_path, correct_answer, \
                video, segment, question = unpack_QAs(QA_data)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": final_question},
                ],
            }
        ]

        with torch.inference_mode():
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        print(output_text)
        answers.append({
            "video": video,
            "segment": segment,
            "question": question,
            # "options": options, 
            # "task_class": task_class, 
            "correct_answer": correct_answer,
            "model_answer": output_text[0]           
        })

    except Exception as e:
        print(f"Error encountered at idx {idx}: {e}")
        answers.append({
            "video": video,
            "segment": segment,
            "question": question,
            # "options": options, 
            # "task_class": task_class, 
            "correct_answer": correct_answer,
            "model_answer": '<error_processing>'           
        })
with open(answers_output_path, 'w', encoding='utf-8') as f:
    json.dump(answers, f, indent=4, ensure_ascii=False)
  