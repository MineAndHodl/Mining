"""Rebuild the published board page from fresh ranked_board.json + the template.
The weekly pick-run calls this after writing rich rationales into ranked_board.json,
then republishes output/the-board.html to the board artifact URL."""
import json, os, sys
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"..","output")
def build(ranked_json=None):
    ranked=ranked_json or json.load(open(os.path.join(HERE,"..","data","ranked_board.json")))
    tmpl=open(os.path.join(OUT,"the-board-template.html")).read()
    # slim the payload to what the page renders — internal/heavy fields stay in ranked_board.json
    DROP = {"contrib", "intel_signals", "thesis", "score_mult", "verdict_note", "grade_raw", "role_mult",
            "inj_lean", "intel_lean", "intel_hard", "inj_note", "offseason", "confidence", "shop_adv",
            "our_num_raw", "score_legacy"}
    def slim(x):
        if isinstance(x, dict): return {k: slim(v) for k, v in x.items() if k not in DROP}
        if isinstance(x, list): return [slim(v) for v in x]
        return x
    page = slim(ranked); page.pop("pool_summary", None)
    html=tmpl.replace("/*DATA*/{}", json.dumps(page, separators=(",", ":")))
    open(os.path.join(OUT,"the-board.html"),"w").write(html)
    return os.path.join(OUT,"the-board.html")
if __name__=="__main__":
    print("built:",build())
