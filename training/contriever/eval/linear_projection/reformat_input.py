import json
import os

def reformat_json(input_file, output_file):

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
    if content.startswith('['):
        data = json.loads(content)
    else:
        data = []
        for line in content.split('\n'):
            if line.strip():
                data.append(json.loads(line.strip()))
    
    for item in data:
        item['input_negative_ctxs'] = item['ctxs']

        for ctx in item['input_negative_ctxs']:
            if 'score' in ctx and isinstance(ctx['score'], str):
                ctx['score'] = float(ctx['score'])
                
        item['ctxs'] = []
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Successfully Processed {len(data)} items")


if __name__ == "__main__":
    input_filename = "two_stage/vanilla_top3/qampari_lin_proj_only_rand_vanilla_top3_unformat_0123.json"
    output_filename = "two_stage/vanilla_top3/qampari_lin_proj_only_rand_vanilla_0123.json"

    input_filename = "two_stage/gold_top0123/qampari_lin_proj_only_rand_gold_ER_unformat_0123.json"
    output_filename = "two_stage/gold_top0123/qampari_lin_proj_only_rand_gold_ER_0123.json"

    input_filename = "two_stage/vanilla_top0123/qampari_lin_proj_only_rand_vanilla_top0123_unformat_0123.json"
    output_filename = "two_stage/vanilla_top0123/qampari_lin_proj_only_rand_vanilla_top0123_25_25_25_25.json"

    reformat_json(input_filename, output_filename)
     #os.remove("two_stage/vanilla_top3/qampari_lin_proj_only_rand_vanilla_top3_unformat_0123.json")
    os.remove("two_stage/vanilla_top0123/qampari_lin_proj_only_rand_vanilla_top0123_unformat_0123.json")
   # os.remove("two_stage/gold_top0123/qampari_lin_proj_only_rand_gold_ER_unformat_0123.json")
