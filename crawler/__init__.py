from gymnasium.envs.registration import register

register(
    id="Crawler-v1",
    entry_point="crawler.crawler_v1:CrawlerV1Env",
    max_episode_steps=200,      # truncation handled by Gymnasium's TimeLimit
)
