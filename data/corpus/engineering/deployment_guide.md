# Deployment Guide

## Overview

This document describes how we deploy services at Meridian Technologies. All production deployments follow a standardized process managed through our CI/CD pipeline. We currently deploy an average of 14 times per week across all services.

## Deployment Process

### Pre-Deployment Checklist

Before initiating any deployment, engineers must verify the following:

1. All unit tests pass with a minimum of 92% code coverage
2. Integration tests have been run against the staging environment
3. The change has been approved by at least two reviewers
4. Database migrations (if any) have been tested in staging
5. Rollback procedures have been documented in the deployment ticket

### Deployment Steps

We use a blue-green deployment strategy for all customer-facing services. The process works as follows:

1. Build the Docker image and push to our internal registry (harbor.meridian.internal)
2. Deploy to the "inactive" environment (blue or green)
3. Run smoke tests against the inactive environment
4. Switch the load balancer to point to the newly deployed environment
5. Monitor error rates for 15 minutes
6. If error rates exceed 0.5%, initiate automatic rollback

### Canary Deployments

For high-risk changes, we use canary deployments where 5% of traffic is routed to the new version for 30 minutes before full rollout. A change is considered high-risk if it modifies payment processing, authentication flows, or database schemas.

## Rollback Process

Rollbacks can be initiated in three ways:

1. **Automatic**: Triggered when error rates exceed 0.5% within the 15-minute monitoring window
2. **Manual**: Any engineer can initiate a rollback through the deployment dashboard
3. **Scheduled**: If a deployment occurs after 4 PM PT on a Friday, it is automatically rolled back and rescheduled for Monday

The rollback process takes approximately 3 minutes and involves switching the load balancer back to the previous environment. Database migrations are not automatically rolled back — these require a separate migration script.

## Environment Configuration

We maintain four environments:

- **Development**: Auto-deploys from feature branches, refreshed daily
- **Staging**: Mirrors production configuration, deployed from the `main` branch
- **Production US**: Primary production environment in us-east-1
- **Production EU**: Secondary production environment in eu-west-1, deployed 2 hours after US to catch issues

All environment configurations are managed through HashiCorp Vault, and secrets are rotated every 90 days. The deployment pipeline is configured to fail if any secret is older than 90 days.

## Monitoring & Alerts

Post-deployment monitoring uses Datadog with the following SLOs:

- API response time p99 < 500ms
- Error rate < 0.1%
- CPU utilization < 70%
- Memory utilization < 80%

Alerts are sent to the #deployments Slack channel and the on-call engineer via PagerDuty.
