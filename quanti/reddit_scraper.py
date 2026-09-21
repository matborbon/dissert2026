"""
Multi-HEI Reddit Scraper for Informal Faculty Evaluation Study
==============================================================
Dissertation: "Beyond the Eval: Exploring Social Media as an Informal
Lens for Faculty Performance in a Digital Age"

Theoretical basis: Granovetter's Strength of Weak Ties (1973)
- Captures cross-subreddit (inter-institutional) bridge users
- Records interaction depth for tie strength operationalization
- Collects all faculty/evaluation-related posts using broad, organic
  language patterns — not just formal evaluation vocabulary

Target subreddits (Philippine HEIs):
  r/Benilde     — College of Saint Benilde
  r/dlsu        — De La Salle University
  r/peyups      — University of the Philippines
  r/AdMU        — Ateneo de Manila University

Usage:
  1. Fill in your Reddit API credentials below (or use environment variables)
  2. pip install praw pandas tqdm
  3. python reddit_scraper.py
"""

import praw
import pandas as pd
import datetime
import time
import os
import logging
from collections import defaultdict
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Reddit API credentials — use environment variables in production
# Never hardcode credentials in the final dissertation appendix
REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID",     "******")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "******")
REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT",    "hei_eval_study/1.0 by matborbon")
REDDIT_USERNAME      = os.getenv("REDDIT_USERNAME",      "******")
REDDIT_PASSWORD      = os.getenv("REDDIT_PASSWORD",      "******")

# Target subreddits — Philippine HEI communities
TARGET_SUBREDDITS = ["Benilde", "dlsu", "peyups", "AdMU"]

# Collection window: 5 academic years
YEARS_BACK = 5
CUTOFF_UTC = int(time.mktime(
    (datetime.datetime.now() - datetime.timedelta(days=YEARS_BACK * 365)).timetuple()
))

# Output paths
OUTPUT_POSTS_CSV    = "dataset_posts.csv"
OUTPUT_COMMENTS_CSV = "dataset_comments.csv"
OUTPUT_NETWORK_CSV  = "dataset_network_edges.csv"
OUTPUT_USERS_CSV    = "dataset_users.csv"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("scraper.log"), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# KEYWORD STRATEGY
# ─────────────────────────────────────────────
# Two-tier approach:
#   TIER 1 — Formal evaluation language (original queries)
#   TIER 2 — Organic student language (colloquial, informal)
# This ensures we don't miss genuine evaluative discourse that
# avoids formal vocabulary — critical for weak tie detection
# since peripheral/bridge users often use casual language.

SEARCH_QUERIES = [
    # ── TIER 1: Formal evaluation vocabulary ──────────────────────
    "professor feedback",
    "teacher evaluation",
    "faculty evaluation",
    "instructor review",
    "course evaluation",
    "teaching effectiveness",
    "student evaluation",
    "SET score",
    "rate my professor",
    "professor review",

    # ── TIER 2: Organic student language ─────────────────────────
    # Recommendation/avoidance framing
    "prof recommendation",
    "avoid professor",
    "best professor",
    "worst professor",
    "good teacher",
    "bad teacher",
    "easy prof",
    "hard prof",
    "strict professor",
    "chill professor",
    "prof pabor",           # Filipino: "professor who is lenient/favoring"
    "prof terror",          # Filipino: "scary/strict professor"
    "prof galing",          # Filipino: "skilled/excellent professor"
    "magandang prof",       # Filipino: "good professor"

    # Teaching quality framing
    "professor explaining",
    "prof doesn't explain",
    "prof spoon feeds",
    "good lecturer",
    "bad lecturer",
    "professor accessible",
    "professor approachable",
    "professor rude",
    "professor dismissive",
    "prof pasaway",         # Filipino: "difficult/troublesome prof"

    # Grading/fairness framing
    "professor grading",
    "unfair grading",
    "fair professor",
    "biased professor",
    "professor gives",
    "professor fails",
    "professor curve",
    "professor recit",      # Philippine HEI: recitation culture
    "professor attendance",
    "cut policy",
    "professor lates",

    # Workload/difficulty framing
    "professor requirements",
    "professor outputs",
    "professor deadlines",
    "heavy workload prof",
    "prof madaming work",   # Filipino: "prof who gives a lot of work"
    "professor majors",
    "professor thesis",
    "professor research",

    # Personal experience/narrative framing
    "my professor",
    "our professor",
    "yung prof",            # Filipino: "that professor"
    "yung teacher",         # Filipino: "that teacher"
    "sir/ma'am review",
    "prof experience",
    "class experience",
    "subject review",
    "prof complaint",
    "professor problem",
    "may alam kayong prof", # Filipino: "do you know a professor"
    "prof na okay",         # Filipino: "a decent professor"
    "prof horror story",
]

# ─────────────────────────────────────────────
# WEAK TIE OPERATIONALIZATION FIELDS
# ─────────────────────────────────────────────
# Per Granovetter (1973), tie strength = f(time, emotional intensity,
# intimacy, reciprocal services). In Reddit's pseudonymous context we
# operationalize as:
#   - Interaction frequency between user pairs (reply chains)
#   - Thread depth (deeper = stronger tie within thread)
#   - Cross-subreddit activity (same user in multiple HEI subs = weak tie bridge)
#   - Upvote score (proxy for broad weak-tie reach vs. niche strong-tie engagement)

def init_reddit():
    """Initialize and return authenticated Reddit instance."""
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        username=REDDIT_USERNAME,
        password=REDDIT_PASSWORD,
    )
    log.info(f"Authenticated as: {reddit.user.me()}")
    return reddit


def is_evaluation_related(text: str) -> bool:
    """
    Secondary filter: confirm post/comment contains evaluative content
    about faculty, not just any use of the keyword.
    Catches organic posts that use evaluation language incidentally.
    """
    evaluation_signals = [
        # Faculty role signals
        "prof", "professor", "teacher", "instructor", "faculty",
        "sir ", "ma'am", "maam", "doc ", "dr.", "lecturer",
        # Evaluation act signals
        "review", "rate", "feedback", "comment", "opinion",
        "recommend", "suggest", "warn", "avoid", "take",
        # Quality judgment signals
        "good", "bad", "best", "worst", "okay", "ok", "great",
        "terrible", "amazing", "awful", "mediocre", "fair", "unfair",
        # Filipino signals
        "galing", "mahirap", "madali", "terror", "pabor", "okay na",
        "okay siya", "hindi okay", "ayos", "ayaw ko", "gusto ko",
    ]
    text_lower = text.lower()
    return any(signal in text_lower for signal in evaluation_signals)


def scrape_subreddit(reddit, subreddit_name: str):
    """
    Scrape all faculty/evaluation-related posts and comments
    from a single subreddit. Returns (posts, comments, edges) lists.
    """
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []
    comments_data = []
    edges_data = []        # For network/SNA edge list
    seen_post_ids = set()  # Deduplicate across queries

    log.info(f"── Scraping r/{subreddit_name} ──")

    for query in tqdm(SEARCH_QUERIES, desc=f"r/{subreddit_name} queries"):
        try:
            results = subreddit.search(query, sort="new", time_filter="all", limit=None)

            for post in results:
                # Skip posts older than cutoff
                if post.created_utc < CUTOFF_UTC:
                    continue

                # Skip already-seen posts (different queries may return same post)
                if post.id in seen_post_ids:
                    continue

                # Secondary evaluation relevance filter
                combined_text = f"{post.title} {post.selftext}"
                if not is_evaluation_related(combined_text):
                    continue

                seen_post_ids.add(post.id)
                post_author = post.author.name if post.author else "deleted"

                # ── Post record ──────────────────────────────────
                posts_data.append({
                    "subreddit":        subreddit_name,
                    "post_id":          post.id,
                    "post_title":       post.title,
                    "post_body":        post.selftext[:2000],  # truncate very long posts
                    "post_author":      post_author,
                    "post_score":       post.score,
                    "post_upvote_ratio": post.upvote_ratio,
                    "post_url":         post.url,
                    "post_num_comments": post.num_comments,
                    "post_created_utc": datetime.datetime.utcfromtimestamp(
                                            post.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "post_flair":       post.link_flair_text,
                    "matched_query":    query,
                })

                # ── Add post-to-subreddit edge (for cross-sub analysis) ──
                edges_data.append({
                    "source":           post_author,
                    "target":           f"r/{subreddit_name}",
                    "edge_type":        "post_to_subreddit",
                    "weight":           post.score,
                    "post_id":          post.id,
                    "subreddit":        subreddit_name,
                    "timestamp":        datetime.datetime.utcfromtimestamp(
                                            post.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
                })

                # ── Collect comments ─────────────────────────────
                try:
                    post.comments.replace_more(limit=0)  # limit=0 avoids rate-limit spikes
                except Exception as e:
                    log.warning(f"replace_more failed on {post.id}: {e}")
                    continue

                for comment in post.comments.list():
                    if isinstance(comment, praw.models.MoreComments):
                        continue
                    if comment.created_utc < CUTOFF_UTC:
                        continue

                    comment_author = comment.author.name if comment.author else "deleted"

                    # Determine thread depth (weak tie signal)
                    # depth = number of "t1_" prefixes in parent chain
                    depth = comment.parent_id.count("t1_")

                    comments_data.append({
                        "subreddit":         subreddit_name,
                        "post_id":           post.id,
                        "post_title":        post.title,
                        "comment_id":        comment.id,
                        "comment_body":      comment.body[:1000],
                        "comment_author":    comment_author,
                        "comment_score":     comment.score,
                        "comment_parent_id": comment.parent_id,
                        "comment_depth":     depth,
                        "comment_created_utc": datetime.datetime.utcfromtimestamp(
                                                comment.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
                        # Tie strength proxy: depth + score
                        # High score + low depth = broad weak-tie reach
                        # Low score + high depth = niche strong-tie exchange
                        "tie_strength_proxy": comment.score / (depth + 1),
                    })

                    # ── Add user-to-user reply edges ─────────────
                    # These are the core edges for SNA
                    # Parent is either post author (depth=0) or another commenter
                    if comment.parent_id.startswith("t1_"):
                        # Reply to a comment — user-to-user weak/strong tie
                        edges_data.append({
                            "source":    comment_author,
                            "target":    comment.parent_id,  # Will be resolved in post-processing
                            "edge_type": "comment_reply",
                            "weight":    comment.score,
                            "post_id":   post.id,
                            "subreddit": subreddit_name,
                            "depth":     depth,
                            "timestamp": datetime.datetime.utcfromtimestamp(
                                            comment.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
                        })
                    else:
                        # Top-level comment — reply to post author
                        edges_data.append({
                            "source":    comment_author,
                            "target":    post_author,
                            "edge_type": "post_reply",
                            "weight":    comment.score,
                            "post_id":   post.id,
                            "subreddit": subreddit_name,
                            "depth":     0,
                            "timestamp": datetime.datetime.utcfromtimestamp(
                                            comment.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
                        })

                # Polite delay to respect Reddit API rate limits
                time.sleep(0.5)

        except Exception as e:
            log.error(f"Error on query '{query}' in r/{subreddit_name}: {e}")
            time.sleep(2)  # Back off on error
            continue

    log.info(
        f"r/{subreddit_name}: {len(posts_data)} posts, "
        f"{len(comments_data)} comments, {len(edges_data)} edges"
    )
    return posts_data, comments_data, edges_data


def compute_cross_subreddit_users(all_posts: pd.DataFrame, all_comments: pd.DataFrame) -> pd.DataFrame:
    """
    Identify users active in multiple subreddits — these are the
    inter-institutional BRIDGE USERS representing Granovetter's weak ties
    at the cross-community level.

    Returns a DataFrame with each user's subreddit activity profile.
    """
    # Collect all author-subreddit pairs
    post_activity = all_posts[["post_author", "subreddit"]].rename(
        columns={"post_author": "user"})
    comment_activity = all_comments[["comment_author", "subreddit"]].rename(
        columns={"comment_author": "user"})

    activity = pd.concat([post_activity, comment_activity], ignore_index=True)
    activity = activity[activity["user"] != "deleted"]

    # Group by user: count unique subreddits and list them
    user_profile = (
        activity.groupby("user")["subreddit"]
        .agg(
            subreddits_active=lambda x: list(x.unique()),
            num_subreddits=lambda x: x.nunique(),
            total_posts_comments="count",
        )
        .reset_index()
    )

    # Flag bridge users (active in 2+ subreddits = inter-institutional weak tie)
    user_profile["is_bridge_user"] = user_profile["num_subreddits"] > 1

    bridge_count = user_profile["is_bridge_user"].sum()
    log.info(
        f"Bridge users (active in 2+ HEI subreddits): {bridge_count} "
        f"({bridge_count/len(user_profile)*100:.1f}% of all users)"
    )

    return user_profile


def resolve_comment_edges(edges: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    """
    Replace parent comment IDs with actual author names for
    user-to-user edge resolution in SNA.
    """
    # Build comment_id → author lookup
    id_to_author = dict(zip(
        comments["comment_id"],
        comments["comment_author"]
    ))

    def resolve_target(row):
        if row["edge_type"] == "comment_reply":
            # Strip "t1_" prefix to get comment ID
            comment_id = row["target"].replace("t1_", "")
            return id_to_author.get(comment_id, "unknown")
        return row["target"]

    edges["target"] = edges.apply(resolve_target, axis=1)
    return edges


def main():
    log.info("Starting multi-HEI Reddit scraper")
    log.info(f"Target subreddits: {TARGET_SUBREDDITS}")
    log.info(f"Collection window: {YEARS_BACK} years back from today")
    log.info(f"Search queries: {len(SEARCH_QUERIES)} queries")

    reddit = init_reddit()

    all_posts     = []
    all_comments  = []
    all_edges     = []

    # ── Scrape each subreddit ──────────────────────────────────────
    for subreddit_name in TARGET_SUBREDDITS:
        posts, comments, edges = scrape_subreddit(reddit, subreddit_name)
        all_posts.extend(posts)
        all_comments.extend(comments)
        all_edges.extend(edges)
        # Polite pause between subreddits
        log.info(f"Pausing 3s before next subreddit...")
        time.sleep(3)

    # ── Convert to DataFrames ──────────────────────────────────────
    df_posts    = pd.DataFrame(all_posts)
    df_comments = pd.DataFrame(all_comments)
    df_edges    = pd.DataFrame(all_edges)

    # ── Resolve comment-to-user edges for SNA ─────────────────────
    if not df_edges.empty and not df_comments.empty:
        df_edges = resolve_comment_edges(df_edges, df_comments)

    # ── Compute cross-subreddit (weak tie) user profiles ──────────
    if not df_posts.empty and not df_comments.empty:
        df_users = compute_cross_subreddit_users(df_posts, df_comments)
        df_users.to_csv(OUTPUT_USERS_CSV, index=False)
        log.info(f"Saved user profiles → {OUTPUT_USERS_CSV}")

    # ── Save datasets ──────────────────────────────────────────────
    df_posts.to_csv(OUTPUT_POSTS_CSV, index=False)
    df_comments.to_csv(OUTPUT_COMMENTS_CSV, index=False)
    df_edges.to_csv(OUTPUT_NETWORK_CSV, index=False)

    log.info("── Collection Summary ──────────────────────────────")
    log.info(f"Total posts collected    : {len(df_posts)}")
    log.info(f"Total comments collected : {len(df_comments)}")
    log.info(f"Total network edges      : {len(df_edges)}")
    if not df_posts.empty:
        log.info(f"Posts by subreddit:\n{df_posts['subreddit'].value_counts().to_string()}")
    log.info(f"Output files: {OUTPUT_POSTS_CSV}, {OUTPUT_COMMENTS_CSV}, "
             f"{OUTPUT_NETWORK_CSV}, {OUTPUT_USERS_CSV}")
    log.info("Scraping complete.")


if __name__ == "__main__":
    main()
