<?php
foreach (['vendor/autoload.php', '../vendor/autoload.php', '../../autoload.php'] as $autoload) {
    $autoload = __DIR__.'/'.$autoload;
    if (file_exists($autoload)) {
        require $autoload;
        break;
    }
}

unset($autoload);

use SimpleNeuralNetwork\Net;

$net = new Net([2, 2, 1]);
$net->train([[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [1]]);

echo '[0, 0]' . PHP_EOL;
echo $net->getOutput([0, 0])[0] . PHP_EOL;

echo '[0, 1]' . PHP_EOL;
echo $net->getOutput([0, 1])[0] . PHP_EOL;
