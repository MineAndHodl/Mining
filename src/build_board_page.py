"""Rebuild the published board page from fresh ranked_board.json + the template.
The weekly pick-run calls this after writing rich rationales into ranked_board.json,
then republishes output/the-board.html to the board artifact URL."""
import json, os, sys
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"..","output")
def build(ranked_json=None):
    ranked=ranked_json or json.load(open(os.path.join(HERE,"..","data","ranked_board.json")))
    tmpl=open(os.path.join(OUT,"the-board-template.html")).read()
    html=tmpl.replace("/*DATA*/{}", json.dumps(ranked))
    open(os.path.join(OUT,"the-board.html"),"w").write(html)
    return os.path.join(OUT,"the-board.html")
if __name__=="__main__":
    print("built:",build())
