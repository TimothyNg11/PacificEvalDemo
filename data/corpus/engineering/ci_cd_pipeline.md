# CI/CD Pipeline Documentation

## Pipeline Overview

Meridian Technologies uses GitHub Actions for continuous integration and ArgoCD for continuous deployment. Our pipeline processes an average of 156 pull requests per week across all repositories. The median time from PR merge to production deployment is 38 minutes.

## Pipeline Stages

### Stage 1: Code Quality (avg. 2 minutes)

- **Linting**: We run `ruff` for Python and `eslint` for TypeScript
- **Type Checking**: `mypy` for Python (strict mode), `tsc` for TypeScript
- **Security Scanning**: `bandit` for Python, `npm audit` for Node.js dependencies
- **Secret Detection**: `truffleHog` scans for accidentally committed secrets

### Stage 2: Unit Tests (avg. 4 minutes)

- All repositories must maintain a minimum of 92% code coverage
- Tests run in parallel across 8 GitHub Actions runners
- Coverage reports are posted as PR comments
- Coverage drops below 92% block the PR from merging

### Stage 3: Integration Tests (avg. 12 minutes)

- Spins up a test environment using Docker Compose
- Runs API contract tests against all service endpoints
- Validates database migrations against a test PostgreSQL instance
- Tests inter-service communication patterns

### Stage 4: Build & Publish (avg. 3 minutes)

- Builds Docker images using multi-stage builds
- Images are tagged with the git SHA and pushed to Harbor registry
- Container size is validated — images exceeding 500MB fail the build
- SBOM (Software Bill of Materials) is generated and stored

### Stage 5: Deploy to Staging (avg. 5 minutes)

- ArgoCD detects the new image tag and syncs the staging environment
- Automated smoke tests run against staging
- Performance regression tests compare p99 latency against the previous version
- If p99 latency increases by more than 15%, deployment is flagged for review

### Stage 6: Deploy to Production (avg. 8 minutes)

- Requires manual approval for services tagged as "critical" (user-service, payment-service, search-service)
- Uses blue-green deployment strategy
- 15-minute monitoring window with automatic rollback on error rate > 0.5%

## Test Coverage Requirements

| Service | Current Coverage | Minimum Required |
|---------|-----------------|------------------|
| user-service | 94.2% | 92% |
| search-service | 91.8% | 92% |
| document-processor | 93.5% | 92% |
| api-gateway | 89.7% | 85% |
| payment-service | 96.1% | 95% |

Note: The search-service is currently 0.2% below the minimum threshold. A ticket (ENG-4521) has been filed to add missing test cases for the hybrid search fallback logic.

## Deployment Frequency

Over the past quarter, our deployment metrics are:

- **Total deployments**: 184
- **Successful deployments**: 178 (96.7% success rate)
- **Rollbacks**: 6 (3.3%)
- **Mean time to recovery**: 4.2 minutes
- **Lead time for changes**: 38 minutes (from merge to production)

We shipped 47 features in Q3 2024, up from 38 in Q2 2024, representing a 23.7% increase in feature velocity. This improvement is attributed to the parallelization of integration tests implemented in July 2024, which reduced the integration test stage from 22 minutes to 12 minutes.

## Branch Strategy

We use trunk-based development:
- All feature branches are short-lived (target: < 2 days)
- Feature flags are used for incomplete features that need to be merged
- Release branches are cut for major versions only
- Hotfix branches deploy directly to production with expedited review (1 reviewer instead of 2)
