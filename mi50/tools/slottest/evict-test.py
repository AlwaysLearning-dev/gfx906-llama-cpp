#!/usr/bin/env python3
# evict-test.py [port]: (1) image on slot 1 while slot 0 holds a longer conversation,
# (2) a 60k-token conversation displaced by two small side requests, then continued.
import json,sys,time,random,base64,io,urllib.request,threading
P=sys.argv[1] if len(sys.argv)>1 else "8080"; U="http://127.0.0.1:%s/v1/chat/completions"%P
def chat(msgs,slot=None,n=40):
    b={"messages":msgs,"max_tokens":n,"chat_template_kwargs":{"enable_thinking":False},"cache_prompt":True}
    if slot is not None: b["id_slot"]=slot
    t=time.time()
    try:
        r=json.load(urllib.request.urlopen(urllib.request.Request(U,json.dumps(b).encode(),{"Content-Type":"application/json"}),timeout=1500))
        tm=r.get("timings",{}); return dict(ok=True,s=time.time()-t,read=tm.get("prompt_n"),cached=tm.get("cache_n"),text=(r["choices"][0]["message"].get("content") or "").strip()[:50])
    except urllib.error.HTTPError as e: return dict(ok=False,s=time.time()-t,err=e.read()[:150].decode())
random.seed(11); W="valve piston ledger harbor cobalt signal lantern quarry tensor marrow glacier orchard socket ripple anchor tunnel violet copper engine meadow".split()
def text(lines): return "\n".join("L%05d %s."%(i," ".join(random.choice(W) for _ in range(9))) for i in range(lines))
def png():
    from PIL import Image, ImageDraw
    im=Image.new("RGB",(1568,882),(20,120,200)); ImageDraw.Draw(im).rectangle((500,300,1000,600),fill=(230,60,40)); b=io.BytesIO(); im.save(b,"PNG")
    return "data:image/png;base64,"+base64.b64encode(b.getvalue()).decode()
print("== 1. image on slot 1 while slot 0 is longer")
r=chat([{"role":"user","content":text(1400)+"\nReply with the single word ready."}],slot=0); print("   slot 0 filled:",r)
r=chat([{"role":"user","content":[{"type":"image_url","image_url":{"url":png()}},{"type":"text","text":"What colour is the rectangle? One word."}]}],slot=1); print("   image on slot 1:",r)
print("== 2. long conversation displaced by two side requests")
A=[{"role":"user","content":text(int(sys.argv[2]) if len(sys.argv)>2 else 2750)+"\nReply with the single word ready."}]
r=chat(A,slot=0); print("   long conversation first read:",r); A+= [{"role":"assistant","content":r.get("text","ready")}]
side=[None,None]
def go(i): side[i]=chat([{"role":"user","content":"Side request %d: %s\nSay ok."%(i,text(40))}])
th=[threading.Thread(target=go,args=(i,)) for i in (0,1)]; [t.start() for t in th]; [t.join() for t in th]; print("   side requests:",side)
r=chat(A+[{"role":"user","content":"Reply with the single word again."}]); print("   long conversation continued:",r)
print("   RESULT: %s (read %s tokens in %.0f s)"%("restored from cache" if r.get("ok") and (r.get("read") or 0)<2000 else "RE-READ",r.get("read"),r.get("s",0)))
