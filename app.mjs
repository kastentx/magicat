import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-node-gpu'

const filename = process.argv[2]

const base64_encode = file => 
  new Buffer.from(fs.readFileSync(file)).toString('base64')

console.log(`using backend: ${ tf.getBackend() }`)
console.log(`submitted filename: ${ filename }`)