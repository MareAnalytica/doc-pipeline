# api_routes

## PaymentRouter

Handles HTTP routes for payment operations.

### create_payment

```python
@app.post("/api/v1/payments")
async def create_payment(
    request: PaymentCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> PaymentResponse:
```

Creates a new payment transaction. Validates the payment request against the user's wallet balance and processes through the configured payment provider.

### get_payment

```python
@app.get("/api/v1/payments/{payment_id}")
async def get_payment(
    payment_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> PaymentResponse:
```

Retrieves a single payment by its ID. Only the payment owner or an admin can access this endpoint.

## PaymentCreateRequest

```python
class PaymentCreateRequest(BaseModel):
    amount: Decimal
    currency: str = "NGN"
    recipient_account: str
    recipient_bank: str
    description: Optional[str] = None
```

## PaymentResponse

```python
class PaymentResponse(BaseModel):
    id: str
    status: PaymentStatus
    amount: Decimal
    currency: str
    created_at: datetime
    reference: str
```

### list_payments

```python
@app.get("/api/v1/payments")
async def list_payments(
    current_user: User = Depends(get_current_user),
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
) -> List[PaymentResponse]:
```

Lists all payments for the authenticated user with pagination support.
