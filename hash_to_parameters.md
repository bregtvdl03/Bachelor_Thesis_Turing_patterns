# 2d_squares_complex

These simulations were run from `0` to `T = 100.0 / gamma` with 
```python
num_steps = 100000
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.04)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.24)
```

unless specified otherwise.

## 0000

### Parameters

```python
Du      = 1.0
Dv      = 20.0
Pu      = 0.1
Pv      = 0.9
gamma   = 64
kappa   = 50

Dw      = 1.0
delta_u = 1.0
delta_w = 1.0
delta_v = 0.0
k_on    = 20.0
```

### Remarks

- No patterns

## 0001

### Parameters

```python
Du      = 1.0
Dv      = 50.0  # Increased
Pu      = 0.1
Pv      = 0.9
gamma   = 64
kappa   = 50

Dw      = 1.0
delta_u = 1.0
delta_w = 1.0
delta_v = 0.0
k_on    = 20.0
```

### Remarks

- Patterns form
- So higher Dv is required
- Lower gamma

## 0010

### Parameters

```python
Du      = 1.0
Dv      = 20.0
Pu      = 0.1
Pv      = 0.9
gamma   = 6 * 64    # Increased
kappa   = 50

Dw      = 1.0
delta_u = 1.0
delta_w = 1.0
delta_v = 0.0
k_on    = 20.0
```

### Remarks

- No patterns
- Really need higher Dv

## 0011

### Parameters

```python
Du      = 1.0
Dv      = 50.0  # Increased
Pu      = 0.1
Pv      = 0.9
gamma   = 64
kappa   = 50

Dw      = 1.0
delta_u = 1.0
delta_w = 1.0
delta_v = 0.1   # Increased
k_on    = 20.0
```

### Remarks

- Patterns form
- delta_v != 0 does not give a uniform steady state, but patterns do generate

## 0100

### Parameters

```python
Du      = 10.0  # Increased
Dv      = 50.0  # Increased
Pu      = 0.1
Pv      = 0.9
gamma   = 64
kappa   = 50

Dw      = 1.0
delta_u = 1.0
delta_w = 1.0
delta_v = 0.1   # Increased
k_on    = 20.0
```

### Remarks

- No patterns

## 0101

### Parameters

```python
Du      = 10.0  # Increased
Dv      = 200.0 # Increased even more
Pu      = 0.1
Pv      = 0.9
gamma   = 64
kappa   = 50

Dw      = 1.0
delta_u = 1.0
delta_w = 1.0
delta_v = 0.1   # Increased
k_on    = 20.0
```

### Remarks

- Patterns form
- Patterns are larger than previous simulations

## 0110

### Parameters

```python
Du      = 1.0
Dv      = 200.0 # Increased even more
Pu      = 0.1
Pv      = 0.9
gamma   = 64
kappa   = 50

Dw      = 10.0  # Increased
delta_u = 1.0
delta_w = 1.0
delta_v = 0.1   # Increased
k_on    = 20.0
```

### Remarks

- Patterns form
- Patterns are only slightly larger
- Increase of `num_steps` necessary (doubled)

## 0111

### Parameters

```python
Du      = 10.0  # Increased
Dv      = 200.0 # Increased even more
Pu      = 0.1
Pv      = 0.9
gamma   = 64
kappa   = 50

Dw      = 10.0  # Increased
delta_u = 1.0
delta_w = 1.0
delta_v = 0.1   # Increased
k_on    = 20.0
```

### Remarks

- Patterns form
- Patterns are larger (expected)
- No increased `num_steps` needed