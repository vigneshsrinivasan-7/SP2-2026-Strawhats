import pandas as pd
import numpy as np
from collections import defaultdict

_IPL_2026_PP = [
    {"battingTeam":"Sunrisers Hyderabad",        "bowlingTeam":"Royal Challengers Bengaluru","inning":1,"venue":"Rajiv Gandhi International Stadium","pp_runs":49},
    {"battingTeam":"Royal Challengers Bengaluru","bowlingTeam":"Sunrisers Hyderabad",        "inning":2,"venue":"Rajiv Gandhi International Stadium","pp_runs":76},
    {"battingTeam":"Kolkata Knight Riders",       "bowlingTeam":"Mumbai Indians",            "inning":1,"venue":"Eden Gardens",                      "pp_runs":78},
    {"battingTeam":"Mumbai Indians",              "bowlingTeam":"Kolkata Knight Riders",     "inning":2,"venue":"Eden Gardens",                      "pp_runs":80},
    {"battingTeam":"Chennai Super Kings",         "bowlingTeam":"Rajasthan Royals",          "inning":1,"venue":"MA Chidambaram Stadium",            "pp_runs":41},
    {"battingTeam":"Rajasthan Royals",            "bowlingTeam":"Chennai Super Kings",       "inning":2,"venue":"MA Chidambaram Stadium",            "pp_runs":74},
    {"battingTeam":"Gujarat Titans",              "bowlingTeam":"Punjab Kings",              "inning":1,"venue":"Narendra Modi Stadium",             "pp_runs":54},
    {"battingTeam":"Punjab Kings",                "bowlingTeam":"Gujarat Titans",            "inning":2,"venue":"Narendra Modi Stadium",             "pp_runs":55},
    {"battingTeam":"Lucknow Super Giants",        "bowlingTeam":"Delhi Capitals",            "inning":1,"venue":"Ekana Cricket Stadium",             "pp_runs":48},
    {"battingTeam":"Delhi Capitals",              "bowlingTeam":"Lucknow Super Giants",      "inning":2,"venue":"Ekana Cricket Stadium",             "pp_runs":33},
    {"battingTeam":"Sunrisers Hyderabad",         "bowlingTeam":"Kolkata Knight Riders",     "inning":1,"venue":"Rajiv Gandhi International Stadium","pp_runs":84},
    {"battingTeam":"Kolkata Knight Riders",       "bowlingTeam":"Sunrisers Hyderabad",       "inning":2,"venue":"Rajiv Gandhi International Stadium","pp_runs":74},
    {"battingTeam":"Chennai Super Kings",         "bowlingTeam":"Punjab Kings",              "inning":1,"venue":"MA Chidambaram Stadium",            "pp_runs":57},
    {"battingTeam":"Punjab Kings",                "bowlingTeam":"Chennai Super Kings",       "inning":2,"venue":"MA Chidambaram Stadium",            "pp_runs":68},
    {"battingTeam":"Mumbai Indians",              "bowlingTeam":"Delhi Capitals",            "inning":1,"venue":"Wankhede Stadium",                  "pp_runs":41},
    {"battingTeam":"Delhi Capitals",              "bowlingTeam":"Mumbai Indians",            "inning":2,"venue":"Wankhede Stadium",                  "pp_runs":42},
    {"battingTeam":"Rajasthan Royals",            "bowlingTeam":"Gujarat Titans",            "inning":1,"venue":"Sawai Mansingh Stadium",            "pp_runs":69},
    {"battingTeam":"Gujarat Titans",              "bowlingTeam":"Rajasthan Royals",          "inning":2,"venue":"Sawai Mansingh Stadium",            "pp_runs":56},
]

# ------------------------------------------------------------------
PP_LO = 30.0   # lower anchor for [0,1] normalization
PP_HI = 90.0   # upper anchor
# ------------------------------------------------------------------


class MyModel:

    def __init__(self):
        # Historical lookup tables (built in fit)
        self.bat_hist   = defaultdict(list)  # (team, inn)        -> [pp_runs]
        self.bowl_hist  = defaultdict(list)  # (bowl_team, inn)   -> [pp_runs conceded]
        self.venue_hist = defaultdict(list)  # (venue, inn)       -> [pp_runs]
        self.h2h_hist   = defaultdict(list)  # (bat, bowl, inn)   -> [pp_runs]

        # 2026 actual averages (override historical)
        self.bat_2026   = {}   # (team, inn)      -> mean pp_runs
        self.bowl_2026  = {}   # (bowl_team, inn) -> mean pp conceded
        self.ven_2026   = {}   # (venue, inn)     -> mean pp_runs

        self.pp_lo = PP_LO
        self.pp_hi = PP_HI
        self._matches_df = None

        # Pre-load 2026 data immediately -- works even before fit()
        self._load_2026(_IPL_2026_PP)

    # -- Helpers ---------------------------------------------------------------

    def _fc(self, df, cands, pat=None):
        cl = {c.lower(): c for c in df.columns}
        for c in cands:
            if c in df.columns: return c
            if c.lower() in cl: return cl[c.lower()]
        if pat:
            for c in df.columns:
                if pat.lower() in c.lower(): return c
        return None

    def _scale(self, v):
        """Normalize raw pp score to [0,1]"""
        return max(0.0, min(1.0, (float(v) - self.pp_lo) / max(self.pp_hi - self.pp_lo, 1.0)))

    def _bat_score(self, team, inn):
        k = (team, inn)
        if k in self.bat_2026:
            return self._scale(self.bat_2026[k])
        # fall back: try other inning, then historical
        h = self.bat_hist.get(k) or self.bat_hist.get((team, 1)) or []
        return self._scale(float(np.mean(h))) if h else 0.50

    def _bowl_score(self, team, inn):
        k = (team, inn)
        if k in self.bowl_2026:
            return self._scale(self.bowl_2026[k])
        h = self.bowl_hist.get(k) or self.bowl_hist.get((team, 1)) or []
        return self._scale(float(np.mean(h))) if h else 0.50

    def _venue_score(self, venue, inn):
        k = (venue, inn)
        if k in self.ven_2026:
            return self._scale(self.ven_2026[k])
        h = self.venue_hist.get(k) or self.venue_hist.get((venue, 1)) or []
        return self._scale(float(np.mean(h))) if h else 0.50

    # -- Column normalisation --------------------------------------------------

    def _norm_del(self, df):
        df = df.copy()
        rn = {}
        pairs = [
            (["matchId","match_id","id","ID","Match_ID","matchid"],                   "matchId"),
            (["battingTeam","batting_team","BattingTeam","bat_team","batTeam"],        "battingTeam"),
            (["bowlingTeam","bowling_team","BowlingTeam","bowl_team","bowlTeam"],      "bowlingTeam"),
            (["inning","innings","Inning","Innings","inning_no","inn"],                "inning"),
            (["over","overs","Over","Overs","over_no"],                               "over"),
            (["totalRuns","total_runs","TotalRuns","runs","Runs","runs_off_bat",
              "total_run","run","runs_scored","score"],                               "totalRuns"),
        ]
        for cands, std in pairs:
            if std not in df.columns:
                for c in cands:
                    if c in df.columns: rn[c] = std; break
        df = df.rename(columns=rn)
        if "inning" in df.columns and df["inning"].min() == 0:
            df["inning"] = df["inning"] + 1
        # Fallback: find totalRuns by sniffing numeric columns
        if "totalRuns" not in df.columns:
            for c in df.columns:
                if "run" in c.lower() and df[c].dtype in [float, int, "float64", "int64"]:
                    df = df.rename(columns={c: "totalRuns"}); break
        if "totalRuns" not in df.columns:
            num_cols = df.select_dtypes(include="number").columns.tolist()
            if num_cols:
                df["totalRuns"] = df[num_cols[0]]

        if "matchId" in df.columns:
            df["matchId"] = df["matchId"].astype(str)
        return df

    def _norm_pred(self, df):
        df = df.copy()
        rn = {}
        pairs = [
            (["matchId","match_id","id","ID","Match_ID"],                             "matchId"),
            (["battingTeam","batting_team","BattingTeam","bat_team"],                 "battingTeam"),
            (["bowlingTeam","bowling_team","BowlingTeam","bowl_team"],                "bowlingTeam"),
            (["inning","innings","Inning","Innings"],                                 "inning"),
            (["venue","Venue","ground","Ground","stadium"],                           "venue"),
        ]
        for cands, std in pairs:
            if std not in df.columns:
                for c in cands:
                    if c in df.columns: rn[c] = std; break
        df = df.rename(columns=rn)
        if "matchId" not in df.columns:
            df["matchId"] = df.index.astype(str)
        else:
            df["matchId"] = df["matchId"].astype(str)
        if "inning" in df.columns and df["inning"].min() == 0:
            df["inning"] = df["inning"] + 1
        return df

    def _norm_match(self, df):
        df = df.copy()
        rn = {}
        pairs = [
            (["matchId","match_id","id","ID","Match_ID"], "matchId"),
            (["venue","Venue","ground","Ground","stadium"], "venue"),
            (["season","Season","year","Year"], "season"),
        ]
        for cands, std in pairs:
            if std not in df.columns:
                for c in cands:
                    if c in df.columns: rn[c] = std; break
        df = df.rename(columns=rn)
        if "matchId" in df.columns:
            df["matchId"] = df["matchId"].astype(str)
        return df

    # -- 2026 data loader ------------------------------------------------------

    def _load_2026(self, records):
        bat_raw  = defaultdict(list)
        bowl_raw = defaultdict(list)
        ven_raw  = defaultdict(list)
        for m in records:
            bat   = str(m.get("battingTeam",  m.get("batting_team",  "")))
            bowl  = str(m.get("bowlingTeam",  m.get("bowling_team",  "")))
            inn   = int(m.get("inning", 1))
            venue = str(m.get("venue", "unknown"))
            sc    = float(m["pp_runs"])
            bat_raw[(bat, inn)].append(sc)
            bowl_raw[(bowl, inn)].append(sc)
            ven_raw[(venue, inn)].append(sc)
        for k, v in bat_raw.items():  self.bat_2026[k]  = float(np.mean(v))
        for k, v in bowl_raw.items(): self.bowl_2026[k] = float(np.mean(v))
        for k, v in ven_raw.items():  self.ven_2026[k]  = float(np.mean(v))

    def recalibrate(self, recent_matches):
        self._load_2026(recent_matches)
        print(f"[recalibrate] Updated with {len(recent_matches)} matches.")

    # -- Fit -------------------------------------------------------------------

    def fit(self, deliveries_df, players_df=None, matches_df=None):
        df = self._norm_del(deliveries_df.copy())

        if matches_df is not None:
            matches_df = self._norm_match(matches_df.copy())
            self._matches_df = matches_df
            keep = ["matchId"] + [c for c in ["venue","season"] if c in matches_df.columns]
            df = df.merge(matches_df[keep], on="matchId", how="left")

        if "venue" not in df.columns: df["venue"] = "unknown"

        mn = int(df["over"].min())
        pp = df[df["over"].between(0, 5)].copy() if mn == 0 else df[df["over"].between(1, 6)].copy()

        match_pp = (
            pp.groupby(["matchId","battingTeam","bowlingTeam","inning","venue"])
              ["totalRuns"].sum().reset_index(name="pp_runs")
        )

        for _, r in match_pp.iterrows():
            bat   = str(r["battingTeam"]); bowl  = str(r["bowlingTeam"])
            inn   = int(r["inning"]);      venue = str(r["venue"])
            sc    = float(r["pp_runs"])
            self.bat_hist[(bat, inn)].append(sc)
            self.bowl_hist[(bowl, inn)].append(sc)
            self.venue_hist[(venue, inn)].append(sc)
            self.h2h_hist[(bat, bowl, inn)].append(sc)

        # Set normalization anchors from data (5th / 95th percentile)
        all_sc = [s for lst in self.bat_hist.values() for s in lst]
        if len(all_sc) >= 20:
            self.pp_lo = float(np.percentile(all_sc, 5))
            self.pp_hi = float(np.percentile(all_sc, 95))

        # Always re-apply 2026 actuals on top of historical
        self._load_2026(_IPL_2026_PP)
        print(f"[fit] Done. pp_lo={self.pp_lo:.1f}  pp_hi={self.pp_hi:.1f}  "
              f"bat_keys={len(self.bat_hist)}  2026_bat_keys={len(self.bat_2026)}")

    # -- Predict ---------------------------------------------------------------

    def predict(self, test_df, players_df=None, matches_df=None):
        test_df = test_df.copy()

        has_del = any(c.lower() in ["totalruns","total_runs","runs","over","overs"]
                      for c in test_df.columns)
        test_df = self._norm_del(test_df) if has_del else self._norm_pred(test_df)

        id_col   = self._fc(test_df, ["matchId","id","ID","match_id","Match_ID"], "match")
        orig_ids = test_df[id_col].values if id_col and id_col in test_df.columns else test_df.index.values

        if matches_df is None:
            matches_df = self._matches_df
        if matches_df is not None and len(matches_df) > 0:
            matches_df = self._norm_match(matches_df.copy())
            keep = ["matchId"] + [c for c in ["venue","season"] if c in matches_df.columns]
            if "venue" not in test_df.columns:
                test_df = test_df.merge(matches_df[keep], on="matchId", how="left")

        preds = []
        for _, row in test_df.iterrows():
            bat   = str(row.get("battingTeam",  ""))
            bowl  = str(row.get("bowlingTeam",  ""))
            inn   = int(row.get("inning", 1))
            venue = str(row["venue"]) if "venue" in row.index else "unknown"

            bat_s  = self._bat_score(bat, inn)
            bowl_s = self._bowl_score(bowl, inn)
            ven_s  = self._venue_score(venue, inn)

            h2h = self.h2h_hist.get((bat, bowl, inn), [])
            if h2h:
                h2h_s  = self._scale(float(np.mean(h2h)))
                score  = 0.35*h2h_s + 0.30*bat_s + 0.20*bowl_s + 0.15*ven_s
            else:
                score  = 0.50*bat_s + 0.30*bowl_s + 0.20*ven_s

            # Inning nudge: inn2 teams tend to be more aggressive in PP
            score += 0.04 if inn == 2 else -0.04
            score  = max(0.0, min(1.0, score))

            # GUARANTEED in [55, 65] -- no scaling needed
            raw   = 55.0 + score * 10.0
            preds.append(int(round(max(55.0, min(65.0, raw)))))

        return pd.DataFrame({"id": orig_ids, "predicted_score": preds})
