import requests, json, os, time
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wiki_committees")
H = {"User-Agent": "stat236-class-project/1.0 (educational)"}
API = "https://en.wikipedia.org/w/api.php"
pages = {"Agriculture":"United States House Committee on Agriculture","Appropriations":"United States House Committee on Appropriations","Armed Services":"United States House Committee on Armed Services","Budget":"United States House Committee on the Budget","Education and the Workforce":"United States House Committee on Education and the Workforce","Energy and Commerce":"United States House Committee on Energy and Commerce","Ethics":"United States House Committee on Ethics","Financial Services":"United States House Committee on Financial Services","Foreign Affairs":"United States House Committee on Foreign Affairs","Homeland Security":"United States House Committee on Homeland Security","House Administration":"United States House Committee on House Administration","Judiciary":"United States House Committee on the Judiciary","Natural Resources":"United States House Committee on Natural Resources","Oversight and Accountability":"United States House Committee on Oversight and Accountability","Rules":"United States House Committee on Rules","Science, Space, and Technology":"United States House Committee on Science, Space, and Technology","Small Business":"United States House Committee on Small Business","Transportation and Infrastructure":"United States House Committee on Transportation and Infrastructure","Veterans' Affairs":"United States House Committee on Veterans' Affairs","Ways and Means":"United States House Committee on Ways and Means","Intelligence":"United States House Permanent Select Committee on Intelligence"}
ts = "2024-10-01T00:00:00Z"; meta = {}
for k, t in pages.items():
    for _ in range(6):
        resp = requests.get(API, headers=H, params=dict(action="query", prop="revisions", titles=t, rvlimit=1, rvdir="older", rvstart=ts, rvprop="ids|timestamp|content", rvslots="main", format="json", redirects=1, formatversion=2))
        try: r = resp.json(); break
        except Exception: time.sleep(10)
    time.sleep(2)
    p = r["query"]["pages"][0]; rev = p["revisions"][0]; txt = rev["slots"]["main"]["content"]
    open(os.path.join(OUT, k.replace(" ","_").replace(",","").replace("'","")+".wiki"), "w", encoding="utf-8").write(txt)
    meta[k] = dict(title=p["title"], oldid=rev["revid"], timestamp=rev["timestamp"], url=f"https://en.wikipedia.org/w/index.php?title={p['title'].replace(' ','_')}&oldid={rev['revid']}")
    print(k, meta[k]["timestamp"], len(txt))
json.dump(meta, open(os.path.join(OUT,"meta.json"),"w"), indent=1)
