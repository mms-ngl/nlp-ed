# Event Detection

## Event schema

Each event can be part of one of the following classes: Sentiment, Scenario, Change, Possession, Action.

| Event class | IOB label tags               | 
|:------------|:-----------------------------|
| Sentiment   | `B-SENTIMENT, I-SENTIMENT`   |
| Scenario    | `B-SCENARIO, I-SCENARIO`     |
| Change      | `B-CHANGE, I-CHANGE`         |
| Possession  | `B-POSESSION, I-POSESSION` |
| Action      | `B-ACTION, I-ACTION`         |
| -           | `O`                          |

A label prefixed with `B-` ("begin") indicates that the corresponding token is the first of an event trigger with the given class.
If the event trigger is composed of multiple tokens, the following labels of the event trigger will be prefixed with `I-` ("inside").
If the token is not part of an event trigger, it will be labelled with `O` ("outside").

