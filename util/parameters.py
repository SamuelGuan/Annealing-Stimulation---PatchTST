optimized_parameters = {
    'sequence_len':[96,128,196,256,512],
    'output_len':[16,32,48,64],
    'patch_size':[8,16,32,48,64],
    'beta':[0.999,0.99,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5],
    'drop_last':[True,False],
    'learning_rate':[0.00001,0.0001,0.0002,0.00025,0.0003,0.00035,0.0005,0.0007],
    'bias':[False]
}

annealingSimulation_parameters = {
    'start_temperature':100,
    'end_temperature':10,
    'cooling_strategy':'loss',
    'optimized_parameters':optimized_parameters,
}
