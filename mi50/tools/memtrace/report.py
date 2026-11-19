import sys,re,subprocess,collections
def load(p):
    st={}; cur=None
    for l in open(p):
        if l.startswith('live tracked'): tot=int(l.split()[-1]); continue
        m=re.match(r'STACK live=(\d+) blocks=(\d+) allocated_total=(\d+)',l)
        if m: cur=[int(m[1]),int(m[2]),int(m[3]),[]]; continue
        if cur is not None and l.startswith('  '):
            cur[3].append(l.strip())
            st[tuple(cur[3])]=cur
    return tot,{tuple(v[3]):v for v in st.values()}
t1,a=load(sys.argv[1]); t2,b=load(sys.argv[2])
print('live tracked: %.0f MiB -> %.0f MiB'%(t1/2**20,t2/2**20))
cache={}
def sym(fr):
    m=re.match(r'(\S+)\+0x([0-9a-f]+)',fr)
    if not m: return fr
    k=(m[1],m[2])
    if k not in cache:
        try: o=subprocess.run(['addr2line','-f','-C','-e',m[1],'0x'+m[2]],capture_output=True,text=True).stdout.split('\n')[0]
        except Exception: o='?'
        cache[k]=o
    return '%s  [%s]'%(cache[k][:110],m[1].split('/')[-1])
rows=sorted(((v[0]-a.get(k,[0])[0],v,k) for k,v in b.items()),key=lambda x:-x[0])
for d,v,k in rows[:int(sys.argv[3]) if len(sys.argv)>3 else 6]:
    print('\ngrowth %+.0f MiB  (live now %.0f MiB in %d blocks, avg %.0f KiB; ever allocated %.0f MiB)'%(d/2**20,v[0]/2**20,v[1],v[0]/max(1,v[1])/1024,v[2]/2**20))
    for fr in k[:12]: print('   ',sym(fr))
