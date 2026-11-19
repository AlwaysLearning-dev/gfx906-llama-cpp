import json,sys,time,urllib.request,subprocess
pid=subprocess.check_output(['pgrep','-x','llama-server']).split()[0].decode()   # run on the server's host
def mem():
    d={}
    for l in open('/proc/%s/status'%pid):
        k,v=l.split(':',1)
        if k in('RssAnon','RssShmem','VmSwap'): d[k]=int(v.split()[0])//1024
    return d['RssAnon']+d['VmSwap'], d['RssShmem']
def req(msgs,n,slot=0,think=False):
    body={'model':'x','messages':msgs,'max_tokens':n,'id_slot':slot,'temperature':0.6,'chat_template_kwargs':{'enable_thinking':think},'cache_prompt':True}
    r=urllib.request.Request('http://127.0.0.1:'+__import__('os').environ.get('PORT','8080')+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
    j=json.load(urllib.request.urlopen(r,timeout=900)); return j['choices'][0]['message'].get('content') or '', j['usage'], j.get('timings',{})
def step(name,msgs,n,**k):
    a0,p0=mem(); t=time.time(); c,u,tm=req(msgs,n,**k); time.sleep(2); a1,p1=mem()
    print('%-34s prompt %6d (read %6s) wrote %5d  own %5d -> %5d (%+5d MiB, %s KiB per written token)  pinned %5d -> %5d  %.0fs'%(name,u['prompt_tokens'],tm.get('prompt_n','?'),u['completion_tokens'],a0,a1,a1-a0,('%.0f'%((a1-a0)*1024/max(1,u['completion_tokens']))),p0,p1,time.time()-t),flush=True)
    return c
mode=sys.argv[1]
if mode in ('gen1','gen3'):
    for i in range(1 if mode=='gen1' else 3):
        step('new chat, long reply',[{'role':'user','content':'Topic %s %d. Write a very long, detailed technical essay about the history of sailing ships. Do not stop early.'%(mode,i)}],2500)
if mode=='gen':
    for i in range(4):
        step('new chat %d, long reply'%i,[{'role':'user','content':'Topic %d. Write a very long, detailed technical essay about the history of sailing ships. Do not stop early.'%i}],2500)
    m=[{'role':'user','content':'Write a long essay about rivers.'}]
    for i in range(3):
        c=step('same chat, turn %d, long reply'%i,m,2500); m+= [{'role':'assistant','content':c},{'role':'user','content':'Continue with much more detail.'}]
if mode=='read':
    filler=' '.join('The quick brown fox %d jumps over the lazy dog.'%i for i in range(1500))
    for i in range(3):
        step('new chat %d, long prompt, short reply'%i,[{'role':'user','content':'Run %d. %s\nReply with the single word OK.'%(i,filler)}],20)
