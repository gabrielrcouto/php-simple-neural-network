<?php
namespace SimpleNeuralNetwork;

class Net
{
    const ALPHA = 0.9;
    const ETA = 0.5;
    const SMALLWT = 0.5;

    protected $cpuCores = 4;
    protected $desiredError;

    protected $inputNeurons = 0;
    protected $hiddenNeurons = 0;
    protected $outputNeurons = 0;

    protected $weightInputHidden;
    protected $weightHiddenOutput;

    public function __construct($neuronsPerLayer, $desiredError = 0.001)
    {
        $this->inputNeurons = $neuronsPerLayer[0];
        $this->hiddenNeurons = $neuronsPerLayer[1];
        $this->outputNeurons = $neuronsPerLayer[2];
        $this->desiredError = $desiredError;
    }

    public function getOutput($inputs)
    {
        array_unshift($inputs, 0.0);

        $sumHidden = array_fill(0, $this->hiddenNeurons + 1, 0.0);
        $sumOutput = array_fill(0, $this->outputNeurons + 1, 0.0);

        $hiddenNeuronValue = array_fill(0, $this->hiddenNeurons + 1, 0.0);
        $outputNeuronValue = array_fill(0, $this->outputNeurons + 1, 0.0);

        for ($j = 1; $j <= $this->hiddenNeurons; $j++) {
            $sumHidden[$j] = $this->weightInputHidden[0][$j];

            for ($i = 1; $i <= $this->inputNeurons; $i++) {
                $sumHidden[$j] += $inputs[$i] * $this->weightInputHidden[$i][$j];
            }

            $hiddenNeuronValue[$j] = 1.0 / (1.0 + exp(-$sumHidden[$j]));
        }

        for ($k = 1; $k <=  $this->outputNeurons; $k++) {
            $sumOutput[$k] = $this->weightHiddenOutput[0][$k];

            for ($j = 1; $j <=  $this->hiddenNeurons; $j++) {
                $sumOutput[$k] += $hiddenNeuronValue[$j] * $this->weightHiddenOutput[$j][$k];
            }

            $outputNeuronValue[$k] = 1.0 / (1.0 + exp(-$sumOutput[$k]));
        }

        array_shift($outputNeuronValue);

        return $outputNeuronValue;
    }

    protected function rando()
    {
        return rand() / (getrandmax() - 1);
    }

    public function train($inputsCollection, $outputsCollection)
    {
        $deltaOutput = array_fill(0, $this->outputNeurons + 1, 0.0);
        $deltaHidden = array_fill(0, $this->hiddenNeurons + 1, 0.0);

        $deltaWeightInputHidden = array_fill(0, $this->inputNeurons + 1, array_fill(0, $this->hiddenNeurons + 1, 0.0));
        $deltaWeightHiddenOutput = array_fill(0, $this->hiddenNeurons + 1, array_fill(0, $this->outputNeurons + 1, 0.0));

        $this->weightInputHidden = array_fill(0, $this->inputNeurons + 1, array_fill(0, $this->hiddenNeurons + 1, 2.0 * ($this->rando() - 0.5) * self::SMALLWT));
        $this->weightHiddenOutput = array_fill(0, $this->hiddenNeurons + 1, array_fill(0, $this->outputNeurons + 1, 0.0));

        $sumHidden = array_fill(0, $this->hiddenNeurons + 1, 0.0);
        $sumOutput = array_fill(0, $this->outputNeurons + 1, 0.0);

        $sumWeightHiddenOutput = array_fill(0, $this->hiddenNeurons + 1, 0.0);

        $hiddenNeuronValue = array_fill(0, $this->hiddenNeurons + 1, 0.0);
        $outputNeuronValue = array_fill(0, $this->outputNeurons + 1, 0.0);

        echo 'Arrays created' . PHP_EOL;

        $sizeOfInputsCollection = count($inputsCollection);

        for ($key = 0; $key < $sizeOfInputsCollection; $key++) {
            array_unshift($inputsCollection[$key], 0.0);
            array_unshift($outputsCollection[$key], 0.0);
        }

        for ($epoch = 0; $epoch < 1000000; $epoch++) {
            $error = 0.0;

            echo 'Epoch ' . $epoch . PHP_EOL;

            $forks = 0;

            for ($key = 0; $key < $sizeOfInputsCollection; $key++) {
                $time = microtime(true);

                $inputs = $inputsCollection[$key];
                $outputs = $outputsCollection[$key];

                for ($j = 1; $j <= $this->hiddenNeurons; $j++) {
                    $sumHidden[$j] = $this->weightInputHidden[0][$j];

                    for ($i = 1; $i <= $this->inputNeurons; $i++) {
                        $sumHidden[$j] += $inputs[$i] * $this->weightInputHidden[$i][$j];
                    }

                    $hiddenNeuronValue[$j] = 1.0 / (1.0 + exp(-$sumHidden[$j]));
                }

                for ($k = 1; $k <= $this->outputNeurons; $k++) {
                    $sumOutput[$k] = $this->weightHiddenOutput[0][$k];

                    for ($j = 1; $j <= $this->hiddenNeurons; $j++) {
                        $sumOutput[$k] += $hiddenNeuronValue[$j] * $this->weightHiddenOutput[$j][$k];
                    }

                    $outputNeuronValue[$k] = 1.0 / (1.0 + exp(-$sumOutput[$k]));

                    // Error SSE
                    $error += 0.5 * ($outputs[$k] - $outputNeuronValue[$k]) * ($outputs[$k] - $outputNeuronValue[$k]);
                    //DeltaO SSE
                    $deltaOutput[$k] = ($outputs[$k] - $outputNeuronValue[$k]) * $outputNeuronValue[$k] * (1.0 - $outputNeuronValue[$k]);
                }

                for ($j = 1; $j <= $this->hiddenNeurons; $j++) {
                    $sumWeightHiddenOutput[$j] = 0.0;

                    for ($k = 1; $k <= $this->outputNeurons; $k++) {
                        $sumWeightHiddenOutput[$j] += $this->weightHiddenOutput[$j][$k] * $deltaOutput[$k];
                    }

                    $deltaHidden[$j] = $sumWeightHiddenOutput[$j] * $hiddenNeuronValue[$j] * (1.0 - $hiddenNeuronValue[$j]);
                }

                for ($j = 1; $j <= $this->hiddenNeurons; $j++) {
                    $deltaWeightInputHidden[0][$j] = self::ETA * $deltaHidden[$j] + self::ALPHA * $deltaWeightInputHidden[0][$j];
                    $this->weightInputHidden[0][$j] += $deltaWeightInputHidden[0][$j];

                    $preCalc = self::ETA * $deltaHidden[$j];

                    for ($i = 1; $i <= $this->inputNeurons; $i++) {
                        $deltaWeightInputHidden[$i][$j] = ($inputs[$i] * $preCalc) + (self::ALPHA * $deltaWeightInputHidden[$i][$j]);
                        $this->weightInputHidden[$i][$j] += $deltaWeightInputHidden[$i][$j];
                    }
                }

                for ($k = 1; $k <= $this->outputNeurons; $k++) {
                    $deltaWeightHiddenOutput[0][$k] = self::ETA * $deltaOutput[$k] + self::ALPHA * $deltaWeightHiddenOutput[0][$k];
                    $this->weightHiddenOutput[0][$k] += $deltaWeightHiddenOutput[0][$k];

                    for ($j = 1; $j <= $this->hiddenNeurons; $j++) {
                        $deltaWeightHiddenOutput[$j][$k] = self::ETA * $hiddenNeuronValue[$j] * $deltaOutput[$k] + self::ALPHA * $deltaWeightHiddenOutput[$j][$k];
                        $this->weightHiddenOutput[$j][$k] += $deltaWeightHiddenOutput[$j][$k];
                    }
                }

                echo '.';
            }

            echo 'Error: ' . $error . PHP_EOL;

            if ($error < $this->desiredError) {
                break;
            }
        }
    }
}
