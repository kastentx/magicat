import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-node-gpu'

const filename = process.argv[2]

console.log(`using backend: ${ tf.getBackend() }`)
console.log(`submitted filename: ${ filename }`)