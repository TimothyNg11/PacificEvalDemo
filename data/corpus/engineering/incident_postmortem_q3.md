# Incident Postmortem: Q3 2024 Search Outage

## Incident Summary

On September 14, 2024, at 2:47 PM PT, the Search Service experienced a complete outage lasting 47 minutes. During this period, all search functionality was unavailable, affecting approximately 12,400 active users. The outage resulted in an estimated revenue impact of $185,000 due to interrupted enterprise customer workflows and SLA credits.

## Timeline

- **2:47 PM**: Automated monitoring detected Search Service error rate spike to 100%. PagerDuty alert fired.
- **2:52 PM**: On-call engineer (Sarah Chen) acknowledged the alert and began investigation.
- **2:58 PM**: Root cause identified — a database migration script dropped an index on the `document_embeddings` table during a routine maintenance window that was incorrectly scheduled to overlap with peak hours.
- **3:05 PM**: We broke prod. The migration was supposed to run at 2 AM but the cron schedule was set to PM instead of AM. The missing index caused all vector search queries to fall back to sequential scan, overwhelming the database.
- **3:12 PM**: Attempted to recreate the index, but the database was under too much load to build the index concurrently.
- **3:18 PM**: Made the decision to failover to the EU database replica, which still had the index.
- **3:24 PM**: Traffic routed to EU replica. Search functionality partially restored with higher latency (avg 340ms vs normal 42ms).
- **3:34 PM**: Index rebuild completed on primary US database.
- **3:34 PM**: Traffic routed back to US primary. Full functionality restored.
- **3:34 PM**: Incident declared resolved. Total downtime: 47 minutes.

## Root Cause

The root cause was a human error in scheduling a database maintenance script. The migration script (`migrate_v2_embeddings.py`) was configured with the cron expression `0 14 * * *` instead of `0 2 * * *`. This caused the index rebuild to execute at 2 PM PT during peak usage instead of 2 AM PT during the maintenance window.

The migration script included `DROP INDEX CONCURRENTLY` followed by `CREATE INDEX CONCURRENTLY`, but the `CREATE INDEX` step failed because the database connection pool was exhausted by the flood of sequential-scan queries.

## Customer Impact

- 12,400 users experienced search failures during the 47-minute window
- 3 enterprise customers (Acme Corp, GlobalTech Industries, and Pinnacle Solutions) triggered SLA breach notifications
- The sales team reported that the Pinnacle Solutions deal (worth $340,000 ARR) was delayed by two weeks as a direct result of the outage, as it occurred during a live product demo
- Customer support received 847 tickets related to the outage

## Remediation

### Immediate Actions (Completed)
1. Fixed the cron schedule to run at 2 AM PT
2. Added a pre-flight check that prevents migrations from running during peak hours (9 AM - 6 PM PT)
3. Added database load monitoring that blocks migrations when query latency exceeds 100ms

### Follow-up Actions (In Progress)
1. Implement blue-green database migrations — build new index before dropping old one
2. Add automated testing for cron schedules in CI
3. Create runbook for database failover to EU replica (target: reduce failover time from 6 minutes to 2 minutes)

## Lessons Learned

1. Never schedule destructive database operations during business hours, even with CONCURRENTLY flag
2. Our EU failover process took too long — need to automate it
3. The monitoring gap between error detection (2:47) and root cause identification (2:58) was 11 minutes. We need better diagnostic dashboards for database-related issues.
