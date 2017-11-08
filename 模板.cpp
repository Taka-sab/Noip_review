分块入门专题
#include<bits/stdc++.h>
using namespace std;
const int maxn = 500010;
int n,m,num,q,x,com,y,z,belong[maxn],block,l[maxn],r[maxn],a[maxn],mark[maxn];
int d[maxn];
char s[10];
//num分块的个数
//belong[i]表示i属于哪一块
//block表示块的大小
//l[i]表示i这块的左端点位置
//r[i]表示右端点位置
void debug() {
	for (int i=1; i<=n; i++)
		printf("%d ",a[i]);
	puts("");
}
void build() {
	block=sqrt(n);
	num=n/block;
	if (n%block) num++;
	for (int i=1; i<=num; i++)
		l[i]=(i-1)*block+1,r[i]=i*block;
	for (int i=1; i<=n; i++)
		belong[i]=(i-1)/block+1;
	for (int i=1; i<=n; i++)
		d[belong[i]]+=a[i];
}
void update(int pos,int x) {
	d[belong[pos]]+=x;
	a[pos]+=x;
}
void query(int L,int R) {
	int ans=0;
	if (belong[L]==belong[R]) {
		for (int i=L; i<=R; i++)
			ans+=a[i];
		printf("%d\n",ans);
		return;
	}
	for (int i=L; i<=r[belong[L]]; i++)
		ans+=a[i];
	for (int i=l[belong[R]]; i<=R; i++)
		ans+=a[i];
	for (int i=belong[L]+1; i<belong[R]; i++)
		ans+=d[i];
	printf("%d\n",ans);
}
main() {
	scanf("%d%d",&n,&q);
	for (int i=1; i<=n; i++)
		scanf("%d",&a[i]);
	build();
	for (int i=1; i<=q; i++) {
		scanf("%d%d%d",&com,&x,&y);
		if (com==2)
			query(x,y);
		else
			update(x,y);
	}
}

含区间加法,带标记的分块
#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <cmath>
using namespace std;
typedef long long ll;
const int maxn = 1000005;
int n,q;
int block,num,l[maxn],r[maxn],belong[maxn],mark[maxn];
ll a[maxn],d[maxn];
void init(){
	block = sqrt(n);
	num = n/block;if(n%block) num++;
	for(int i = 1;i<=num;i++) l[i] = (i-1)*block+1;
	for(int i = 1;i<=num;i++) r[i] = i*block;
	for(int i = 1;i<=n;i++) belong[i] = (i-1)/block+1;
	for(int i = 1;i<=n;i++) d[i] = a[i];
	for(int i = 1;i<=num;i++) sort(d+l[i],d+r[i]+1);
}
void update(int L,int R,int x){
	if(belong[L]==belong[R]){
		if(mark[belong[L]]){
			for(int i = l[belong[L]];i<=r[belong[R]];i++) a[i]+=mark[belong[L]];
		}
		mark[belong[L]] = 0;
		for(int i = L;i<=R;i++) a[i]+=x;
		for(int i = l[belong[L]];i<=r[belong[R]];i++) d[i]=a[i];
		sort(d+l[belong[L]],d+r[belong[L]]+1);
		return;
	}
	update(L,r[belong[L]],x);
	update(l[belong[R]],R,x);
	for(int i = belong[L]+1;i<=belong[R]-1;i++) mark[i]+=x;
}
int query(int L,int R,int C){
	int ret = 0;
	if(belong[L]==belong[R]){
		for(int i = L;i<=R;i++) if(a[i]+mark[belong[L]]>=C) ret++;
		return ret;
	}
	ret+= query(L,r[belong[L]],C)+query(l[belong[R]],R,C);
	for(int i = belong[L]+1;i<=belong[R]-1;i++){
		int l1=l[i],r1=r[i],temp=0;
		while(l1<=r1)
		{
			int mid=(l1+r1)/2;
			if (d[mid]+mark[i]>=C)
				r1=mid-1,temp=r[i]-mid+1;
			else
				l1=mid+1;
		}
		ret+=temp;
	}
	return ret;
}
int main(){
	scanf("%d%d",&n,&q);
	for(int i = 1;i<=n;i++){
		scanf("%lld",a+i);
	}
	init();
	char s[10];
	int x,y,z;
	for (int i=1;i<=q;i++)
	{
		scanf("%s%d%d%d",s,&x,&y,&z);
		if (s[0]=='A')
			cout<<query(x,y,z)<<endl;
		else
			update(x,y,z);
	}
	return 0;
}

莫队
#include <stdio.h>
#include <iostream>
#include <algorithm>
using namespace std;
const int maxn  = 50005;
struct quer{
	int lef,righ;
	int id;
}quers[200005];
inline bool operator<(const quer &q1,const quer &q2){
	return q1.lef/500==q2.lef/500?q1.righ<q2.righ:q1.lef/500<q2.lef/500;
}
int a[maxn];
int cnt[maxn];
int ans,aans[maxn];
inline void add(int i){
	cnt[a[i]]++;
	if(cnt[a[i]]==1) ans++;
}
inline void remove(int i){
	cnt[a[i]]--;
	if(cnt[a[i]]==0) ans--;
}
int main(){
	int n;
	cin>>n;
	for(int i  = 1;i<=n;i++){
		cin>>a[i];
	}
	int q;
	cin>>q;
	for(int i = 1;i<=q;i++){
		quers[i].id = i;
	   cin>>quers[i].lef>>quers[i].righ;
	}
	sort(quers+1,quers+1+q);
	int L = 1,R = 0;
	for(int i = 1;i<=q;i++){
		int qL = quers[i].lef,qR = quers[i].righ;
		while(R<qR) add(++R);
	        while(L<qL) remove(L++);
	        while(R>qR) remove(R--);
	        while(L>qL) add(--L);
	    aans[quers[i].id] = ans;
	}
	for(int i = 1;i<=q;i++) cout<<aans[i]<<endl;
	return 0;
}

链表 vector 再加二分 可以A很多很难的题!!!

并查集的路径压缩
inline int find(int a){
	if(father[a]==a){
		return a;
	}else{
		return father[a]  = find(father[a]);
	}
}

欧拉筛法
#include <cstring>
using namespace std;
int prime[1100000],primesize,phi[11000000];
bool isprime[11000000];
void getlist(int listsize)
{
    memset(isprime,1,sizeof(isprime));
    isprime[1]=false;
    for(int i=2;i<=listsize;i++)
    {
        if(isprime[i])prime[++primesize]=i;
         for(int j=1;j<=primesize&&i*prime[j]<=listsize;j++)
         {
            isprime[i*prime[j]]=false;
            if(i%prime[j]==0)break;// == 0 !!
        }
    }
}


大数求mod

for(int i = 0;i<=n;i++){
		cin>>a[i]+1;
		int len = strlen(a[i]+1);
		int j;
		int flag = 0;
		if(a[i][1]=='-') {
		j = 2;
		flag = 1;
		}
		else j = 1;
		for(;j<=len;j++){
			b[i] = (b[i]*10+(a[i][j]-'0'));
			b[i] %= mod;
		}
		if(flag){
			b[i] = -b[i];
		}
	}

对拍代码
@echo off
fc test1.txt test2.txt
pause
或
int main()
{
    while(1)
    {
        system("rand.exe");
        system("暴力.exe");
        system("hotel.exe");
        if(system("fc 暴力.txt hotel.txt ")) //fc是比较文本，如果文本相同，返回1
          while(1);
    }
    return 0;
}
//注意 include stdlib

部分位置固定后的括号匹配
//括号匹配
我们可以贪心的进行从后向前如果这个括号没有
指定成右括号而且他是左括号合法，则标记为左
括号 否则为右括号。跑完之后看看是否合法
#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cstdlib>
using namespace std;
const int maxn = 1000007;
int p = 0;
int z[maxn],a[maxn],n,m;
int main()
{
  freopen("program.in","r",stdin);
  freopen("program.out","w",stdout);
  scanf("%d",&n);
  for (int i = 1; i<=n;i++)
    scanf("%d",&a[i]);
  scanf("%d",&m);
  int x;
  for (int i = 1; i<=m;i++)
  {
    scanf("%d",&x);
    if (a[x]>0) a[x] = -a[x];
  }
  for (int x = n; x>=1;x--)
  {
    if (a[x]>0) {
      if (z[p] == -a[x]) p--;
        else { a[x] = -a[x]; p++; z[p]= a[x];}
    } else
    {
      p++;
      z[p]=a[x];
    }  
  }
  if (p==0)
  {
    for (int i = 1; i<=n;i++)
    {
       if (a[i]>0) printf("+%d ",a[i]);
         else printf("%d ",a[i]);
        }
    }
    else printf("NO\n");
  return 0;
}


链表
#include <stdio.h>
#include <iostream>
using namespace std;
const int maxn = 3e5+1;
struct node {
    int color;
    int index;
} colors[maxn];
int head[maxn],nex[maxn],a[maxn],last[maxn];
int main() {
    int n,m;
    scanf("%d%d",&n,&m);
    for(int i = 1; i<=n; i++) {
        cin>>colors[i].color;
        colors[i].index = i;
        a[i] = colors[i].color;
        if(last[a[i]]!=0) {
                nex[last[a[i]]] = i;
                //pre[i] = last[a[i]];
                last[a[i]] = i;
            } else {
                head[a[i]] = last[a[i]] = i;
            }
    }
    for(int i = 1; i<=m; i++) {
        int t;
        scanf("%d",&t);
        if(t==1) {
            int l,r,c;scanf("%d%d%d",&l,&r,&c);
            int ans = 0;
            for(int j = head[c];j;j=nex[j]){
                if(colors[j].index<=r&&colors[j].index>=l) ans++;
                else if(colors[j].index>r) break;
            }
            cout<<ans<<endl;
        } else {
            int x;
            scanf("%d",&x);
            if(a[x]==a[x+1]) continue;
            int j;
            for(j = head[a[x]]; j; j=nex[j]) {
                if(colors[j].index==x) {
                    colors[j].index++;
                    break;
                }
            }
            for(j = head[a[x+1]]; j; j=nex[j]) {
                if(colors[j].index==x+1) {
                    colors[j].index--;
                    break;
                }
            }
            swap(a[x],a[x+1]);
        }
    }
    return 0;
}

计算第一个 大于等于a 且是 k 的整数倍的数
int n = (a+k-1)/k*k

LCT & 线段树
#include<iostream>
#include<cstring>
#include<cmath>
#include<algorithm>
#include<cstdio>
#define ll long long
#define maxn 100010//数组千万不要开大了或者开小了
using namespace std;
int a[maxn],n;
int head[maxn],num;
int size[maxn],son[maxn],top[maxn],end[maxn],fa[maxn],dep[maxn],pos[maxn];
int dfsx[maxn],xu=1;
ll sum[maxn<<2],lazy[maxn<<2],bui=1;
struct node{
    int to,from,next;
}e[maxn<<1];
ll read()//这个并没有什么用
{
    int w=1,x=0;char ch=getchar();
    while(ch>'9'||ch<'0'){if(ch=='-')w=-1;ch=getchar();}
    while(ch>='0'&&ch<='9')x=(x<<3)+(x<<1)+ch-'0',ch=getchar();
    return w*x;
}
//以下是线段树代码
void add(int from,int to)//链式前向星存边
{
    num++;
    e[num].to=to;
    e[num].next=head[from];
    head[from]=num;
}
void build(int root,int l,int r)//建线段树
{
    if(l==r)
    {
        sum[root]=a[dfsx[bui++]];//a[dfsx]存的是已经处理好的树链
        return ;
    }
    int mid=(l+r)/2;
    build(root<<1,l,mid);
    build(root<<1|1,mid+1,r);
    sum[root]=sum[root<<1]+sum[root<<1|1];
}
void push(int root,int l,int r)
{
    int mid=(l+r)/2;
    lazy[root<<1]+=lazy[root];
    lazy[root<<1|1]+=lazy[root];
    sum[root<<1]+=lazy[root]*(mid-l+1);
    sum[root<<1|1]+=lazy[root]*(r-mid);
    lazy[root]=0;
    return;
}
void jia(int root,int left,int right,int l,int r,ll k)
{
    if(left>=l&&r>=right)
    {
        lazy[root]+=k;
        sum[root]+=k*(right-left+1);
        return ;
    }
    if(l>right||r<left)return ;
    int mid=(left+right)/2;
    if(lazy[root])push(root,left,right);
    if(l<=mid)jia(root<<1,left,mid,l,r,k);
    if(r>mid)jia(root<<1|1,mid+1,right,l,r,k);
    sum[root]=sum[root<<1]+sum[root<<1|1];
    return ;
}
ll query(int root,int left,int right,int l,int r)
{
    if(left>=l&&right<=r)return sum[root];
    if(l>right||left>r)return 0;
    if(lazy[root])push(root,left,right);
    int mid=(left+right)/2;
    ll a=query(root<<1,left,mid,l,r);
    ll b=query(root<<1|1,mid+1,right,l,r);
    return a+b;
}
//以下是树剖代码（以轻重链为基础）
void dfs1(int x)//第一个深搜，由第一个深搜我们可以得到的是每个点的深度，它的重儿子，以它为节点的子树的节点个数
{
    int maxsize=0;//为了存重儿子
    for(int i=head[x];i;i=e[i].next)
    {
        int u=e[i].to;
        if(u!=fa[x])//因为是双向边，所以不能走父亲节点
        {
            dep[u]=dep[x]+1;
            fa[u]=x;
            size[x]++;
            dfs1(u);
            size[x]+=size[u];
            if(size[u]>maxsize)
            {
                son[x]=u;
                maxsize=size[u];
            }
        }
    }
    return ;
}
void dfs2(int x,int s)//第二个深搜是构造重链，按顺序存下树链
{
    bool f=1;//判断是否有重儿子
    if(son[x])
    {
        f=0;
        top[son[x]]=s;
        dfsx[xu++]=son[x];//重链是以优先每个重儿子相连的
        dfs2(son[x],s);//深搜它的重儿子
    }
    for(int i=head[x];i;i=e[i].next)
    {    
        int u=e[i].to;
        if(u!=fa[x]&&u!=son[x])
        {
            if(f)//如果当前点既不是父节点又不是重儿子就判断它是否是另外一条重链的头
            top[u]=s,f=0;
            else top[u]=u;//有重儿子，它就是顶端
            dfsx[xu++]=u;
            dfs2(u,u);
        }
    }
    end[x]=xu-1;//以x开头的链有多长，方便线段树的维护
}
void update_path(int x,int y,int z)//与lca相似，目的就是维护x，y的最短路径
{
    int fx=top[x],fy=top[y];
    while(fx!=fy)//将x，y所在的重链不断往上跳
    {
        if(dep[fx]>=dep[fy])
        {
            jia(1,1,n,pos[fx],pos[x],z);
            x=fa[fx];
        }
        else
        {
            jia(1,1,n,pos[fy],pos[y],z);
            y=fa[fy];
        }
        fx=top[x],fy=top[y];
}//线段树维护

    if(x!=y)//看当前x，y在树链中的位置
    {
        if(pos[x]<pos[y])
        {
            jia(1,1,n,pos[x],pos[y],z);
        }
        else jia(1,1,n,pos[y],pos[x],z);
    }
    else jia(1,1,n,pos[x],pos[y],z);
}
ll query_path(int x,int y)//同上，只是求值而已
{
    ll ret=0;
    int fx=top[x],fy=top[y];
    while(fx!=fy)
    {
        if(dep[fx]>=dep[fy])
        {
            ret+=query(1,1,n,pos[fx],pos[x]);
            x=fa[fx];
        }
        else
        {
            ret+=query(1,1,n,pos[fy],pos[y]);
            y=fa[fy];
        }
        fx=top[x],fy=top[y];
    }
    if(x!=y)
    {
        if(pos[x]<pos[y])
        {
            ret+=query(1,1,n,pos[x],pos[y]);
        }
        else ret+=query(1,1,n,pos[y],pos[x]);
    }
    else ret+=query(1,1,n,pos[x],pos[y]);
    return ret;
}
void update_subtree(int x,int z)
{
    jia(1,1,n,pos[x],end[x],z);
}
ll query_subtree(int x)
{
    return query(1,1,n,pos[x],end[x]);
}
int main()
{
    int m,r,x,y,z,t;
    ll p;
    cin>>n>>m>>r>>p;
    for(int i=1;i<=n;i++)scanf("%d",&a[i]);
    for(int i=1;i<n;i++)
    {
        scanf("%d%d",&x,&y);
        add(x,y);
        add(y,x);
    }
    fa[r]=r;//开头的预处理
    dep[r]=0;
    dfs1(r);
    top[r]=r;//以根为第一条链
    dfsx[xu++]=r;//入队
    dfs2(r,r);
    for(int i=1;i<xu;i++)
    {
        pos[dfsx[i]]=i;//按依次求出链的顺序入队
    }
    build(1,1,n);
    while(m--)
    {
        scanf("%d",&t);
            if(t==1)
            {
                scanf("%d%d%d",&x,&y,&z);
                update_path(x,y,z);
                continue;
            }
            if(t==2)
            {
                scanf("%d%d",&x,&y);
                printf("%lld\n",query_path(x,y)%p);
                continue;
            }
            if(t==3)
            {
                scanf("%d%d",&x,&z);
                update_subtree(x,z);
                continue;
            }
            if(t==4)
            {    
                scanf("%d",&x);
                printf("%lld\n",query_subtree(x)%p);
                continue;
            }
    }
    return 0;
}

LCT

#include <stdio.h>
#include <iostream>
#include <vector>
using namespace std;
const int maxn = 5e5+4;
int n,m;
int size[maxn],top[maxn],son[maxn],time,fat[maxn],dep[maxn];
int nex[maxn<<1],head[maxn<<1],dis[maxn<<1],q;
inline void connect(int x,int y) {
    nex[++q] = head[x],head[x] = q,dis[q] = y;
    nex[++q] = head[y],head[y] = q,dis[q] = x;
}
void dfs(int a,int fa) {
    dep[a] = dep[fa]+1;
    fat[a] = fa;
    size[a] = 1;
    son[a] = -1;
    for(int i = head[a]; i; i=nex[i]) {
        int to = dis[i];
        if(to!=fa) {
            dfs(to,a);
            size[a]+=size[to];
            if(son[a]==-1||size[to]>size[son[a]]) {
                son[a] = to;
            }
        }
    }
}
void dfs2(int a,int t) {

    top[a] = t;
    if(son[a]==-1) return;
    dfs2(son[a],t);
    for(int i = head[a]; i; i=nex[i]) {
        int to = dis[i];
        if(to!=fat[a]&&to!=son[a]) {
            dfs2(to,to);
        }
    }
}
int main() {
    int s;
    scanf("%d%d%d",&n,&m,&s);
    for(int i = 1; i<=n-1; i++) {
        int x,y;
        scanf("%d%d",&x,&y);
        connect(x,y);
    }
    dfs(s,s);
    dfs2(s,s);
    for(int i=1; i<=m; ++i) {
        int x,y;
        scanf("%d%d",&x,&y);
        while(top[x]!=top[y]) {
            if(dep[top[x]]>=dep[top[y]])x=fat[top[x]];
            else y=fat[top[y]];
        }
        printf("%d\n",dep[x]<dep[y]?x:y);
    }
    return 0;
}
预处理相同元素下次出现的位置
    for(int i=1;i<=n;i++)last[i]=p+1;
    for(int i=1;i<=p;i++)
        a[i]=read();
    for(int i=p;i;i--)
    {
        nxt[i]=last[a[i]];
        last[a[i]]=i;
    }

树状数组 求 逆序对
#include <stdio.h>
#include <algorithm>
#include <iostream>
#define maxn 40005
using namespace std;
struct node{
	int val,pos;
}nodes[40005];
bool operator<(const node &n1,const node &n2){
	return n1.val<n2.val;
}
int a[maxn],c[maxn];
inline int lowbit(int x){return x&(-x);}
inline int getSum(int x){int ret = 0;while(x>0){ret+=c[x];x-=lowbit(x);}return ret;}
inline void add(int x){	while(x<=maxn){c[x]++;	x+=lowbit(x);}}
int main()
{
	int n;
	cin>>n;
	for(int i = 1;i<=n;i++){
		cin>>nodes[i].val;
		nodes[i].pos = i;
	}
	//li san hua
	sort(nodes+1,nodes+1+n);
	for(int i = 1;i<=n;i++){
		a[nodes[i].pos] = i;
	}
	//end li san hua
	long long ans = 0;
	for(int i = 1;i<=n;i++){
		add(a[i]);
		ans+=i-getSum(a[i]);
	}
	cout<<ans;
	return 0;
}


hash
hash1 = 131;
for(int i = 0; i<len; i++) {
hash1 = (hash1*19260817+1l*s[i]);
}

string判断相等
bool b3 = (strcmp(p1, "Hello World") == 0); // true
读入一行cin.getline
这是因为，cin>>counter;
后，你有一个回车键停留在缓冲区中了，当使用cin.getline(a,100);
读到这个回车符，就认为输入结束了，所以，看起来没有起作用。
cin>>counter;
cin.get();   //加上这一句。
删除指定位置字符 who[strlen(who)-1]='\0';

高精度 乘 低精度 + 除 低精度
struct BigInteger{
	int a[maxn];
	int length;
};
void operator*=( BigInteger &bg1,const int muti){
	int x =0;
	for(int i =1;i<=bg1.length;i++){
		bg1.a[i] = bg1.a[i]*muti+x;//不要忘了进位
		x = bg1.a[i]/10;
		bg1.a[i]%=10;
	}
	while(x){
		bg1.a[++bg1.length] = x%10;
		x/=10;
	}
}
void operator/=(BigInteger &bg,const int div){
	int x = 0;
	for(int i = bg.length;i>=1;i--){
		bg.a[i] = x*10+bg.a[i];
		x = bg.a[i]%div;//没有10
		bg.a[i] /= div;
	}
	while(!bg.a[bg.length]&&bg.length)bg.length--;
}
void output(BigInteger &bg){
	for(int i = bg.length;i>=1;i--){
		cout<<bg.a[i];
	}
        if(!bg.length)cout<<0;
}

线段树重点部分
#include <stdio.h>
#include <iostream>
#define inf 1e9
using namespace std;
int mins[4000003];
int a[1000005];
int addMark[4000003];
void buildTree(int node,int left,int right){
	if(left>=right){
		mins[node] = a[left];
		return;
	}
	int mid = (left+right)>>1;
	buildTree(node*2,left,mid);
	buildTree(node*2+1,mid+1,right);
	mins[node] = min(mins[node*2],mins[node*2+1]);
}

void pushMark(int node){
	if(addMark[node]){
		addMark[node*2+1]+= addMark[node];addMark[node*2]+=addMark[node];
		mins[node*2+1]+=addMark[node];mins[node*2]+=addMark[node];
		addMark[node] = 0;
	}
}
int query(int node,int start,int end,int qstart,int qend){
	if(qend<start||qstart>end){
		return inf;
	}
	if(qstart<=start&&end<=qend){ //注意了 这里是 搜索的区间 包含在了 被查询区间 !!!错过一次!!
		return mins[node];
	}
	pushMark(node);
	int mid = (start+end)>>1;
	return min(query(node*2,start,mid,qstart,qend),query(node*2+1,mid+1,end,qstart,qend));
}
void update(int node,int start,int end,int qstart,int qend,int dt){
	if(qend<start||qstart>end){
		return;
	}
	if(qstart<=start&&end<=qend)
	{
		addMark[node]+=dt;
		mins[node]+=dt;
		return;
	}else{
		pushMark(node);//这里向下传递 是保证下面的数据都是最新的 这个地方很重要 不要忘了
		int mid = (start+end)>>1;
		update(node*2,start,mid,qstart,qend,dt);
		update(node*2+1,mid+1,end,qstart,qend,dt);
		mins[node] = min(mins[node*2],mins[node*2+1]);
	}
}
int main(){
	int n,m;
	cin>>n>>m;
	for(int i = 1;i<=n;i++)
	{
		cin>>a[i];
	}
	buildTree(1,1,n);
	for(int i = 1;i<=m;i++){
		int need,from,to;
		cin>>need>>from>>to;
		if(query(1,1,n,from,to)>=need){
			update(1,1,n,from,to,-need);
		}else{
			cout<<-1<<endl;
			cout<<i;
			return 0;
		}
	}
	cout<<0<<endl;
	return 0;
}

ST表
#include <iostream>
#include <cstdio>
#include <cmath>
using namespace std;
int f[202020][20];
int n,m,a,b;
int query(int l,int r)
{
    int k=log(r-l+1)/log(2);
    return max(f[l][k],f[r-(1<<k)+1][k]);
}
int main()
{
    scanf("%d%d",&n,&m);
    for(int i=1; i<=n; i++)
        scanf("%d",&f[i][0]);
    for(int i=1; i<=20; i++)
        for(int j=1; j+(1<<i)-1<=n; j++){
		
            f[j][i]=max(f[j][i-1],f[j+(1<<(i-1))][i-1]);
		
			}
    for(int i=1; i<=m; i++)
    {
        scanf("%d%d",&a,&b);
        printf("%d\n",query(a,b));
    }
    return 0;
}


Tarjan 割点(去掉后不连通的点）
#include<cstdio>
#include<iostream>
#include<bits/stdc++.h>
using namespace std;
const int N=100001;
int n,m,cnt,ans;
int h[N],dfn[N],low[N],fa[N];
//dfn[u]表示节点u被访问的时间，
//low[u]表示u及u的子树中所有结点能到达的结点中dfn最小的结点的时间
//fa[u]表示u的祖先结点
bool cut[N];
struct node{
    int v;
    int next;
}e[2*N];//注意无向边相当于正反两条有向边，不然RE
void Add(int u,int v){
    cnt++;
    e[cnt].next=h[u],
    e[cnt].v=v,
    h[u]=cnt;
}
void Tarjan(int p){
    int rd=0;
    dfn[p]=low[p]=++cnt;
    for(int i=h[p];i;i=e[i].next){
        int v=e[i].v;
        if(!dfn[v]){
            fa[v]=fa[p];Tarjan(v);//合并，或者不用fa数组每次递归是带上fa也行
            low[p]=min(low[p],low[v]);
            if(low[v]>=dfn[p]&&p!=fa[p])cut[p]=true;
        //非根且子树能达到的dfn最小的结点的时间>=自己的时间时
            //说明他的子树中最早能访问到的结点都比他后访问，只要不为根就一定是割点（注意根例外）
            if(p==fa[p])rd++;
        }
        low[p]=min(low[p],dfn[v]);//把p及其子树可以达到的dfn最小结点更新
    }
    if(p==fa[p]&&rd>=2)cut[fa[p]]=true;//入度>=2且为根的结点，因为一棵树的根一删不管有几棵子树肯定都不连通了
}
int main(){
    int n,m;
    scanf("%d%d",&n,&m);
    for(int i=1;i<=m;i++){
        int x,y;
        scanf("%d%d",&x,&y);
        Add(x,y);Add(y,x);
    }
    memset(dfn,0,sizeof(dfn));
    for(int i=1;i<=n;i++)fa[i]=i;//初始化每个结点的父亲结点都为其本身
    for(int i=1;i<=n;i++)
        if(!dfn[i])Tarjan(i);
    for(int i=1;i<=n;i++)
    if(cut[i])ans++;
    printf("%d\n",ans);
    for(int i=1;i<=n;i++)
    if(cut[i])printf("%d ",i);//看题，别在输出格式上翻车
    return 0;
}
/*和求强连通分量的区别主要是在low[u]=min(low[u],dfn[v])能否改成low[u]=min(low[u],low[v])上，




void pr(int k)      //求k的质因子  
{  
    num = 0;  
    for (int i = 2 ; i * i <= k ; i++)  
    {  
        if (k % i == 0)  
        {  
            p[num++] = i;  
            while (k % i == 0)  
                k /= i;  
        }  
    }  
    if (k > 1)  
        p[num++] = k;  
}  
二分图匹配(可以自己向自己建边!)
#include <stdio.h>
#include <cstring>
#include <iostream>
#include <vector>
#define maxn 1005
using namespace std;
vector<int> G[maxn];
int visit[maxn];
int match[maxn];
int dfs(int x){
	for(int i = 0;i<G[x].size();i++){
		int to = G[x][i];
		if(!visit[to]){
			visit[to] = 1;
			if(!match[to]||dfs(match[to])){
				match[to] =  x;
				return 1;
			}
		}
	}
}
int main(){
	int q = 1;
	int n,m,e;
	cin>>n>>m>>e;
	for(int i = 1;i<=e;i++){
		int u,v;
		cin>>u>>v;
		if(u>n||v>m) continue;
		G[u].push_back(v);//只需要建单向边
	}
	int ans = 0;
	for(int i = 1;i<=n;i++){
		memset(visit,0,sizeof(visit));
		if(dfs(i)) ans++;
	}
	cout<<ans;
	return 0;
}

判断是不是割点(求了强连通后直接调用）
  for(int i=2;i<=n;++i)
    {
        int v=father[i];
        if(v==1)
        rootson++;/*统计根节点子树的个数，根节点的子树个数>=2,就是割点*/
        else{
            if(low[i]>=dfn[v])/*割点的条件*/
            is_cut[v]=true;
        }
    }
    if(rootson>1)
    is_cut[1]=true;

Tarjan强连通 并求出入度为0的强联通分量个数
#include <stdio.h>
#include <iostream>
#include <cstring>
#include <vector>
using namespace std;
const int maxn  = 205;
int low[maxn],dfn[maxn],time;
struct edge{
	int from,to;
}edges[maxn*maxn];
vector<int> G[maxn];
class stack{
	int a[maxn];
	int visit[maxn];
	int rare;
	public:
	void init(){
		memset(visit,0,sizeof(visit));
		 rare  = 0;
	}
	public:
	void push(int x){
		visit[x] = 1;
		a[rare++] = x;
	}
	public:
	int pop(){
		int ret = a[--rare];
		visit[ret] = 0;
		return ret;
	}
	public:
	int isIn(int x){
		return visit[x];
	}
};

stack s;
int f[maxn];
int scc;
void tarjan(int x) {
	low[x]=dfn[x] = ++time;
	s.push(x);
	for(int i = 0;i<G[x].size();i++){
		if(!dfn[G[x][i]]){
			tarjan(G[x][i]);
			low[x] = min(low[x],low[G[x][i]]);
		}else if(s.isIn(G[x][i])) low[x] = min(low[x],dfn[G[x][i]]);
	}
	if(dfn[x]==low[x]){
		scc++;
		do{
			 x = s.pop();
			 f[x] = scc;
		}while(dfn[x]!=low[x]);
	}
}
int q;
int ru[maxn],chu[maxn];
void connect(int a,int b){
	q++;
	edges[q].from = a;
	edges[q].to = b;
	G[a].push_back(b);
}
int main() {
	int n;
	s.init();
	cin>>n;
	for(int i = 1;i<=n;i++){
		int x;
		cin>>x;
		while(x){
			connect(i,x);
			cin>>x;
		}
	}
	for(int i = 1;i<=n;i++){
		if(!dfn[i]) tarjan(i);
	}
	for(int i = 1;i<=q;i++){
		edge &ed = edges[i];
		if(f[ed.from]!=f[ed.to]){//f表示所属强连通分量编号
			//not in the same scc
			chu[f[ed.from]]++;
			ru[f[ed.to]]++;
		}
	}
	int ans = 0;
	for(int i = 1;i<=scc;i++) if(!ru[i]) ans++;
	cout<<ans;
	return 0;
}

Tarjan强联通
//Low数组是一个标记数组，记录该点所在的强连通子图所在搜索子树的根节点的Dfn值，Dfn数组记录搜索到该点的时间，也就是第几个搜索这个点的
void tarjan(int a){
	low[a] = dfn[a] = ++time;
	s.push(a);
	for(int i = 0;i<G[a].size();i++){
		if(!dfn[G[a][i]]){
			tarjan(G[a][i]);
			low[a] = min(low[a],low[G[a][i]]);
		}else if(s.in(G[a][i])) low[a] = min(low[a],dfn[G[a][i]]);
	}
	printf("(%d,%d,%d)",a,low[a],dfn[a]);
	if(dfn[a] == low[a]){//是一个"末"点
		do{
			a=s.pop();
			printf("%d ",a);
		}while(dfn[a]!=low[a]);
		printf("\n");
	}
}
差分约束
#include <stdio.h>
#include <queue>
#include <string.h>
#include <iostream>
#include <vector>
#define maxn 10005
using namespace std;
typedef long long ll;
struct edge{
    int to,w;
    edge(int to1,int w1){
        to = to1,w = w1;
    }
};
vector<edge> G[maxn];
void addEdge(int a,int b,int c){
    //a to b
    G[a].push_back(edge(b,c));
}
int visit[maxn],flag = false,dis[maxn];
void spfa(int a){
    visit[a] = 1;
    for(int i = 0;i<G[a].size();i++){
        int to = G[a][i].to;
        int c = G[a][i].w;
        if(dis[to]>dis[i]+c){
            if(visit[to]){
                flag = true;
                return;
            }
            dis[to] = dis[i]+c;
            spfa(to);
        }
    }
    visit[a] = 0;//注意这里是要出队的
}
int main(){
    int n,m;
    scanf("%d%d",&n,&m);
    int x,a,b,c;
    for(int i = 1;i<=m;i++){
        scanf("%d%d%d",&x,&a,&b);
        if(x==1){
            scanf("%d",&c);
            addEdge(a,b,-c);
        }else if(x==2){
            scanf("%d",&c);
            addEdge(b,a,c);
        }else if(x==3){
            addEdge(a,b,0);addEdge(b,a,0);
        }
    }
    for(int i = 1;i<=n;i++) dis[i] = 1e9;
    for(int i = 1;i<=n;i++){
        dis[i] = 0;
        spfa(i);
        if(flag) break;
    }
    if(flag){
        cout<<"No";
    }else{
        cout<<"Yes";
    }
    return 0;
}
bfs版spfa  通过入队次数判断
bool spfa(int n) {  
     int u, v;  
   
     while (!q.empty()) q.pop();  
     memset(vis, false, sizeof(vis));  
     memset(in, 0, sizeof(in));  
     fill(d, d + n, oo);  
     
     d[0] = 0; vis[0] = true;  
     q.push(0);  
     
     while (!q.empty()) {  
         u = q.front();  
         vis[u] = false;  
         for (int i = prev[u]; i != -1; i = edge[i].next) {  
             v = edge[i].v;  
             if (d[u] + edge[i].t < d[v]) {  
                 d[v] = d[u] + edge[i].t;  
                 if (!vis[v]) {  
                     in[v] ++;  
                     if (in[v] > n) return true;                //存在一点入队次数大于总顶点数  
                     vis[v] = true;  
                     q.push(v);  
                 }  
             }  
         }  
         
         vis[u] = false;  
         q.pop();  
     }  
   
     return false;  
}  
dfs版SPFA 判环
void spfa(int x){
    f[x]=true;
    for (int i=0;i<a[x].size();i++){
        A e=a[x][i];
        if (dis[e.to]>dis[x]+e.cost){
            if (f[e.to]){//visit
                flag=true;
                return;
            }
            dis[e.to]=dis[x]+e.cost;
            spfa(e.to);
        }
    }
    f[x]=false;//注意出队
    return;
}
简短的EXGCD
void exgcd(int a,int b,int &x,int &y) //求ax+by=1的函数
{
    if (!b) x=1,y=0;
    else
    {
        exgcd(b,a%b,y,x);
        y-=x*(a/b);//kuo hao!!
    }
}
quickPow取模
ll quickPow(ll b,ll p,ll k){
	ll ans = 1;
	b%=k;
	while(p){
		if(p&1){
			ans=ans*b%k;//im
		}
		p>>=1;
		b = b*b%k;
	}
	return ans;
}
求phi表
int p[MAXN];  
void phi(){  
    for(int i=1;i<MAXN;i++) p[i]=i;  
    for(int i=2;i<MAXN;i++){  
        if(p[i]==i){  //因为J从i开始 所以不需要是 p【i】==i-1
            for(int j=i;j<MAXN;j+=i)  
                p[j]-=p[j]/i;  
        }  
    }  
}
nlgn的lis
d[1]=a[1];  
    int len=1;
    for (int i=2;i<=n;i++)
    {
        if (a[i]>=d[len]) d[++len]=a[i];  
        else  
        {
            int j=upper_bound(d+1,d+len+1,a[i])-d;
            d[j]=a[i];
        }
    }
    printf("%d",len);  
