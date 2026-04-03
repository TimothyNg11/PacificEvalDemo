# Product Roadmap — H2 2024

## Overview

This document outlines Meridian Technologies' product priorities for the second half of 2024. Our roadmap is organized around three strategic pillars: Enterprise Readiness, Intelligence Features, and Platform Scalability.

## Pillar 1: Enterprise Readiness

### Advanced Analytics Dashboard (Target: Q4 2024)

**Priority: P0 — Critical**

Real-time usage analytics for enterprise administrators. This feature has been requested by 34 of our top 50 enterprise customers and is a blocker for three deals in the pipeline worth a combined $1.2 million ARR.

Key capabilities:
- User activity tracking (searches per user, documents uploaded)
- Department-level usage breakdowns
- ROI calculator showing time saved vs. manual document search
- Exportable reports in PDF and CSV formats
- Customizable dashboards with drag-and-drop widgets

Dependencies:
- Requires completion of the data warehouse migration (Engineering, ETA: September 2024)
- Needs new telemetry SDK integration (Engineering, ETA: August 2024)

### SSO/SAML Integration (Target: Q3 2024)

**Priority: P0 — Critical**

Enterprise customers require Single Sign-On integration. Currently, 28% of enterprise prospects cite lack of SSO as a reason for delayed procurement. We are implementing SAML 2.0 and OIDC support.

### Audit Logging (Target: Q3 2024)

**Priority: P1 — High**

Comprehensive audit logging for compliance-sensitive customers. Every API call, document access, user login, and admin action will be logged with timestamps and user identity. Logs will be retained for 7 years and exportable to customer SIEM systems.

## Pillar 2: Intelligence Features

### AI-Powered Document Summaries (Target: Q1 2025)

**Priority: P1 — High**

Automatic generation of document summaries using large language models. This feature will process uploaded documents and generate a 3-5 sentence summary, key entities, and suggested tags.

Technical approach:
- Use a hosted LLM (initially GPT-4, with plans to support self-hosted models)
- Process in the existing Document Processing Pipeline
- Cache summaries to avoid re-computation

### Smart Search Suggestions (Target: Q1 2025)

**Priority: P2 — Medium**

AI-powered search query suggestions based on user search history and document corpus analysis. Shows "related searches" and "you might also be interested in" suggestions.

## Pillar 3: Platform Scalability

### API v3 (Target: Q1 2025)

**Priority: P1 — High**

A completely redesigned API with:
- GraphQL support alongside REST
- Webhook support for real-time event notifications
- Improved rate limiting with burst allowances
- Better pagination with cursor-based navigation
- Comprehensive OpenAPI 3.1 documentation

### Multi-Region Data Residency (Target: Q4 2024)

**Priority: P0 — Critical**

Allow customers to choose where their data is stored (US, EU, or APAC). Required for GDPR compliance and to unlock the European market. Data will not cross regional boundaries.

## Resource Allocation

- Enterprise Readiness: 45% of engineering capacity (8 engineers)
- Intelligence Features: 25% of engineering capacity (4 engineers)
- Platform Scalability: 20% of engineering capacity (3 engineers)
- Tech Debt & Reliability: 10% of engineering capacity (2 engineers)

## Risk Factors

The data warehouse migration is on the critical path for the Analytics Dashboard. A delay in the migration would push the Analytics Dashboard to Q1 2025, which could impact three enterprise deals.
