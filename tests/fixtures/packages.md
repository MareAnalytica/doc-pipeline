# service

```go
import "github.com/justpay/backend/internal/justpay/service"
```

Package service provides the core business logic for JustPay payment operations.

## Index

- [type PaymentService](<#PaymentService>)
- [func NewPaymentService](<#NewPaymentService>)
- [func (s *PaymentService) ProcessPayment](<#PaymentService.ProcessPayment>)
- [func (s *PaymentService) GetPaymentStatus](<#PaymentService.GetPaymentStatus>)
- [type TenancyService](<#TenancyService>)

## type PaymentService

PaymentService handles all payment processing logic including bill payments and wallet top-ups.

```go
type PaymentService struct {
    db        *sqlx.DB
    cache     *redis.Client
    provider  PaymentProvider
    notifier  NotificationService
}
```

### func NewPaymentService

NewPaymentService creates a PaymentService with the given dependencies.

```go
func NewPaymentService(db *sqlx.DB, cache *redis.Client, provider PaymentProvider) *PaymentService
```

### func (*PaymentService) ProcessPayment

ProcessPayment executes a payment transaction and returns the result.

```go
func (s *PaymentService) ProcessPayment(ctx context.Context, req PaymentRequest) (PaymentResult, error)
```

### func (*PaymentService) GetPaymentStatus

GetPaymentStatus retrieves the current status of a payment by reference ID.

```go
func (s *PaymentService) GetPaymentStatus(ctx context.Context, referenceID string) (PaymentStatus, error)
```

## type TenancyService

TenancyService manages rental tenancy records and lease agreements.

```go
type TenancyService struct {
    db       *sqlx.DB
    notifier NotificationService
}
```

### func NewTenancyService

```go
func NewTenancyService(db *sqlx.DB, notifier NotificationService) *TenancyService
```

### func (*TenancyService) CreateLease

CreateLease initializes a new tenancy lease agreement between a landlord and tenant.

```go
func (s *TenancyService) CreateLease(ctx context.Context, req CreateLeaseRequest) (*Lease, error)
```

### func (*TenancyService) GetTenantPayments

GetTenantPayments returns the payment history for a specific tenant.

```go
func (s *TenancyService) GetTenantPayments(ctx context.Context, tenantID string) ([]Payment, error)
```
