const fs = require('fs')
const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node-gpu')

const filename = process.argv[2]
const base64_encode = file => 
  new Buffer.from(fs.readFileSync(file)).toString('base64')

console.log(`using backend: ${ tf.getBackend() }`)
console.log(`submitted filename: ${ filename }`)
console.log(`submitted image: ${ base64_encode(filename) }`)