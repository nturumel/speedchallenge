	?߾??	@?߾??	@!?߾??	@	?????T!@?????T!@!?????T!@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?߾??	@?/?$??AGr???@Y?5?;N???*	?????	r@2F
Iterator::ModelQk?w????!=C??ͳR@)?j+?????16???o?J@:Preprocessing2U
Iterator::Model::ParallelMapV2????!????V5@)????1????V5@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceL7?A`???!??aHx?&@)L7?A`???1??aHx?&@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeata??+e??!?g	?/!@)?j+??ݓ?16???o?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??g??s??!???g	-@)??y?):??1??f*??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_vOv?!??????)??_vOv?1??????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?Pk?w???!?"??09@)a2U0*?s?1ȩ?Xy???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?J?4??!????J?-@)a2U0*?S?1ȩ?Xy???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t18.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?????T!@I,+*?f?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?/?$???/?$??!?/?$??      ??!       "      ??!       *      ??!       2	Gr???@Gr???@!Gr???@:      ??!       B      ??!       J	?5?;N????5?;N???!?5?;N???R      ??!       Z	?5?;N????5?;N???!?5?;N???b      ??!       JCPU_ONLYY?????T!@b q,+*?f?V@