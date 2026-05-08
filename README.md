# Posture-Aware Smart-Home Interaction System

This repository contains the implementation of a posture-aware smart-home interaction prototype for contactless smart-home device control. The system combines visual perception, hand-state recognition, motion trend analysis, control intention filtering, and real-device execution through the Mijia API.

The project was developed as a final year project prototype. It supports contactless control of a smart lamp and a smart speaker using upper-body posture, hand landmarks, gesture-state transitions, and short-term motion-window features.

## Project Overview

Many gesture-control systems directly map a recognised hand gesture to a predefined command. This project extends that design by adding action understanding and control intention judgement before command execution.

The implemented system checks whether:

- the user's hand is in a valid interaction region;
- the movement shows a meaningful direction or trend;
- the action is likely to be intentional control;
- the command satisfies stability, cooldown, and session constraints;
- the selected smart-home device can execute the mapped command.

The goal is to demonstrate a complete prototype-stage control pipeline rather than an isolated gesture classifier.

## Main Features

- Upper-body pose detection using MediaPipe Pose
- Hand landmark detection using MediaPipe Hands
- Open/Fist gesture-state recognition for discrete control
- Geometric OK-gesture detection for fine-control mode
- ROI-based hand-region enhancement
- Motion trend analysis using hand and body-reference coordinates
- 20-frame motion-window feature extraction
- 1D-CNN intention classifier for prototype-stage false-trigger filtering
- Rapid arm-drop suppression after operation completion
- Dead-zone filtering, rate limiting, cooldown, and session constraints
- Real smart-home device linkage through the Mijia API
- Smart lamp and smart speaker control

## Supported Device Functions

| Device | Supported Functions |
|---|---|
| Xiaomi Mi Smart LED Desk Lamp 1S | On/off control, brightness adjustment, colour-temperature adjustment |
| Mijia smart speaker | Playback control, volume adjustment, track switching |

## System Pipeline

The system follows a multi-stage control pipeline:

1. **Device context setup**  
   The system connects to the Mijia account, retrieves available devices, and selects the smart lamp and smart speaker for the current session.

2. **Perception layer**  
   Camera frames are processed with MediaPipe Pose and MediaPipe Hands to obtain upper-body landmarks, hand landmarks, hand state, and hand-region information.

3. **Action understanding layer**  
   The system analyses hand trajectory, movement direction, dominant axis, displacement trend, and movement strength.

4. **Intentional and unintentional control recognition**  
   A 20-frame motion window is passed to a 1D-CNN intention classifier. The classifier is combined with rule-based checks to reduce false triggers from natural arm withdrawal and rapid arm dropping.

5. **Control logic**  
   Accepted gestures and movement trends are mapped to discrete or continuous smart-home commands.

6. **API sending and feedback**  
   Commands are sent to real smart-home devices through the Mijia API. Device responses and interface state updates close the interaction loop.

## Repository Structure

```text
.
├── mijia_intent_control_main.py       # Main real-device control program
├── train_intent_model.py              # Intention classifier training script
├── label_any_video.py                 # Semi-automatic video labelling tool
├── collect_data.py                    # Data collection utility
├── record_video.py                    # Video recording utility
├── predictor_core.py                  # Real-time prediction / feature processing core
├── realtime.py                        # Real-time testing or auxiliary script
├── Device.py                          # Device abstraction
├── config.py                          # Local configuration file
├── gesture_controller_simulated.py    # Simulated control version
├── gesture_recognizer.task            # MediaPipe gesture recognizer task file
├── mijiaAPI/                          # Mijia API wrapper
├── demos/                             # Demo materials
└── jsons/                             # Local/private JSON files, not for public release