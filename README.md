# Machine Learning with Arduino: Quadratic Model Fitting for Sensor Data

This project demonstrates the implementation of a machine learning algorithm for quadratic model fitting using an Arduino. It collects data from sensors, trains a model, and performs predictions to control output devices based on the estimated values.

---

## Features

- **Quadratic Model Fitting**: Implements least squares regression for quadratic models.
- **Real-Time Performance Metrics**: Calculates and displays error metrics such as MSE, RMSE, MAE, and R².
- **Data Splitting**: Automatically splits training and testing data for model evaluation.
- **Control Output**: Dynamically adjusts outputs based on predictions.
- **Interactive User Input**: Collects sensor and control values via the Arduino serial monitor.
- **Debugging Mode**: Prints detailed matrix computations and model parameters.

---

## Pin Configuration

| Pin Name         | Pin Number |
|-------------------|------------|
| Sensor Input      | `A0`       |
| Human Input       | `A1`       |
| Control Output    | `10`       |
| Begin Training    | `A2`       |
| Testing Mode PB   | `6`        |
| Model Valid PB    | `5`        |
| Test PB           | `4`        |

---

## State Machine

The system operates based on the following states:

1. **Idle**: Waits for user input to begin training or enter testing mode.
2. **TrainingStart**: Collects training data from sensors and users.
3. **CollectingDataOnEntry**: Increments the data count.
4. **CollectingDataOnStay**: Records sensor and control values.
5. **FitModel**: Performs quadratic regression to fit the model.
6. **CollectX**: Reads new sensor data.
7. **CalculateY**: Predicts control values using the trained model.
8. **UpdateControlOnEntry/Stay**: Updates the control output based on predictions.
9. **CollectTestData**: Collects data for model testing.
10. **EstimateY**: Calculates the predicted control value during testing.
11. **AccumError**: Updates error metrics for performance evaluation.
12. **ShowError**: Displays the performance metrics.

---

## Performance Metrics

- **MSE (Mean Squared Error)**: Average squared difference between actual and predicted values.
- **RMSE (Root Mean Squared Error)**: Square root of MSE, indicating the standard deviation of errors.
- **MAE (Mean Absolute Error)**: Average absolute difference between actual and predicted values.
- **R² (Coefficient of Determination)**: Proportion of variance explained by the model.

---

## Model Training and Testing

The data is divided into training and testing sets based on a specified portion (`TestingPortion`). Matrix operations are used for model fitting, including:

1. **Transpose Matrix**
2. **Matrix Multiplication**
3. **Matrix Inversion**

The weights of the quadratic model (`w`) are calculated using these operations. The testing set is then used to evaluate the model's performance.

---

## Code Overview

### Main Functions

1. **`setup()`**: Initializes pins, begins serial communication, and optionally runs training in debug mode.
2. **`loop()`**: Handles the state machine transitions and operations based on user input and sensor data.
3. **`QuadraticModelFitting()`**: Fits a quadratic model to the training data using matrix operations.
4. **`performanceReport()`**: Calculates and displays performance metrics.
5. **`train_test_split()`**: Splits data into training and testing sets.
6. **`delayG(int time)`**: Implements non-blocking delays.

### Debugging Mode

When `DebugAIMode` is set to `true`, detailed outputs are printed to the serial monitor, including:

- Training data
- Model weights
- Performance metrics

---

## Requirements

- **Hardware**: Arduino (e.g., Uno), sensors, and output devices (e.g., LEDs, motors).
- **Software**: Arduino IDE.

---

## How to Use

1. Connect the components to the Arduino as per the pin configuration.
2. Upload the code to the Arduino using the Arduino IDE.
3. Open the serial monitor and follow the prompts to:
   - Collect training data.
   - Train the model.
   - Perform predictions.
4. Observe the real-time performance metrics displayed in the serial monitor.

---

## Future Enhancements

- Expand the model to handle higher-order polynomials.
- Add support for saving and loading trained models to/from EEPROM.
- Implement advanced error metrics for better evaluation.
- Integrate visualization of real-time predictions and errors.

---

## License

This project is open-source and available under the MIT License. Contributions are welcome!
