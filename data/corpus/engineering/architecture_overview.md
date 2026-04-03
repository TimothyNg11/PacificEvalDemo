# Architecture Overview

## System Design

Meridian Technologies operates a microservices architecture consisting of 23 services running on Kubernetes (EKS) across two AWS regions. The system handles approximately 2.8 million API requests per day with a 99.97% uptime SLA.

## Core Services

### API Gateway

The API Gateway is built on Kong and handles authentication, rate limiting, and request routing. It processes an average of 32,000 requests per minute during peak hours. Rate limits are set to 1,000 requests per minute per API key for standard tier and 10,000 for enterprise tier customers.

### User Service

The User Service manages authentication and user profiles. It uses PostgreSQL 15 as its primary datastore with a read replica for analytics queries. The service maintains approximately 847,000 active user accounts as of Q3 2024.

### Document Processing Pipeline

The document processing pipeline is the core of our product. It consists of three stages:

1. **Ingestion**: Documents are uploaded to S3 and a message is published to SQS. The ingestion service processes documents at a rate of approximately 450 documents per minute.
2. **Processing**: Documents are parsed, chunked, and embedded using a fine-tuned embedding model. Processing takes an average of 2.3 seconds per document.
3. **Indexing**: Processed documents are indexed in our vector store (Pinecone) and metadata is stored in PostgreSQL.

### Search Service

The Search Service handles all retrieval operations. It supports three search modes: vector search, keyword search (BM25), and hybrid search. The service maintains a cache layer using Redis with a 78% cache hit rate, which reduces average query latency from 180ms to 42ms.

## Tech Stack

- **Languages**: Python 3.11 (backend services), TypeScript (frontend, API gateway plugins)
- **Databases**: PostgreSQL 15, Redis 7, Pinecone (vector store)
- **Infrastructure**: AWS (EKS, S3, SQS, RDS), Terraform for IaC
- **Monitoring**: Datadog (metrics/traces), PagerDuty (alerts), Sentry (errors)
- **CI/CD**: GitHub Actions for CI, ArgoCD for continuous deployment

## Scaling Architecture

We use horizontal pod autoscaling (HPA) with the following thresholds:

- Scale up when CPU > 60% for 3 minutes
- Scale down when CPU < 30% for 10 minutes
- Minimum replicas: 3 per service
- Maximum replicas: 50 per service

The Document Processing Pipeline has additional scaling based on SQS queue depth — we add one processing pod per 500 queued documents.

## Data Flow

All inter-service communication happens via gRPC for synchronous calls and SQS for asynchronous operations. We migrated from REST to gRPC in Q2 2024, which reduced inter-service latency by 34% and payload sizes by 62%.

## Security

All data in transit is encrypted using TLS 1.3. Data at rest is encrypted using AES-256. We completed SOC 2 Type II certification in August 2024. PII is stored in a dedicated encrypted database partition with access logged and audited monthly.
