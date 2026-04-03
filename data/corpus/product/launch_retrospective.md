# Launch Retrospective — Q3 2024 Releases

## Overview

This retrospective covers the three major product launches in Q3 2024: Hybrid Search, SSO/SAML Integration, and Audit Logging. We review what shipped versus what was planned, adoption metrics, and lessons learned.

## Release 1: Hybrid Search

### What We Planned
- Combine vector and BM25 keyword search for improved relevance
- Support three search modes: vector-only, keyword-only, and hybrid
- Target 25% improvement in search recall over vector-only

### What We Shipped
- Hybrid search with Reciprocal Rank Fusion (RRF) scoring
- Three search modes as planned, plus an additional "hybrid + rerank" mode using a cross-encoder model
- Achieved 31% improvement in search recall over vector-only (exceeded target)
- Delivered 2 weeks ahead of schedule

### Adoption Metrics (as of September 30)
- 62% of search API calls now use hybrid mode (up from 0% at launch)
- 8% use hybrid+rerank mode
- 22% remain on vector-only
- 8% use keyword-only
- Search-related support tickets decreased by 41% compared to Q2

### What Went Well
- The cross-encoder reranking mode was a late addition (suggested by the ML team) that has become highly valued by enterprise customers
- Thorough A/B testing in staging caught two edge cases that would have caused relevance regressions
- Customer communication plan was effective — 89% of surveyed customers were aware of the feature within 2 weeks

### What Could Be Improved
- We underestimated the latency impact of the reranker — p99 latency for hybrid+rerank is 420ms vs. 180ms for vector-only. Need to optimize.
- Documentation for the new search modes was published 3 days after launch. Should have been day-0.
- No A/B testing of the RRF constant (k=60). Should experiment with different values.

## Release 2: SSO/SAML Integration

### What We Planned
- SAML 2.0 support for enterprise SSO
- Integration with major identity providers (Okta, Azure AD, OneLogin)
- Admin UI for SSO configuration

### What We Shipped
- SAML 2.0 support as planned
- Tested integrations with Okta, Azure AD, OneLogin, and Google Workspace
- Admin UI with step-by-step SSO setup wizard
- Delivered 1 week late due to Azure AD edge cases

### Adoption Metrics
- 34 enterprise customers have configured SSO (39% of enterprise base)
- 12 additional customers have SSO configuration in progress
- SSO-related objections in sales process decreased from 28% to 8%

### What Went Well
- The setup wizard received very positive feedback — average configuration time is 15 minutes
- Zero security incidents related to the SSO implementation

### What Could Be Improved
- Azure AD integration required 3 hotfixes in the first week
- We should have included OIDC support (deferred to Q4)
- Load testing revealed performance issues with SSO token validation at >1,000 concurrent users

## Release 3: Audit Logging

### What We Planned
- Log all API calls, document access, user authentication events
- 7-year retention
- CSV export for compliance teams

### What We Shipped
- Comprehensive audit logging as planned
- 7-year retention with configurable policies
- Export in CSV, JSON, and SIEM-compatible formats (Splunk, Datadog)
- Real-time audit log streaming via webhooks (unplanned addition)

### Adoption Metrics
- 41 enterprise customers have enabled audit logging (47% of enterprise base)
- 6 customers are using the real-time streaming feature
- Audit logging cited as a factor in 4 enterprise deal wins in Q3

### What Went Well
- Real-time streaming was a customer-driven addition that paid off immediately
- Storage costs were 40% lower than estimated due to efficient compression

### What Could Be Improved
- Query performance on large audit logs (>10 million events) is slow. Need to implement partitioning.
- The admin UI for searching audit logs needs filtering improvements

## Overall Lessons Learned

1. **Ship earlier, iterate faster** — All three features benefited from being in customers' hands. Real-world usage revealed improvements we couldn't have anticipated.
2. **Documentation should ship with the feature** — The 3-day delay on Hybrid Search documentation generated 47 support tickets that could have been avoided.
3. **Late additions can be high-value** — Both the reranker mode and audit log streaming were late additions that became key selling points.
4. **Buffer time for identity provider edge cases** — Enterprise integrations always have surprising edge cases. Build in at least 1 extra week for enterprise features.
