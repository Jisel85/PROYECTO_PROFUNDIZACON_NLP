/* global use, db */

use('Project_NLP');

// db.getCollection('judgments').find({});

db.getCollection('judgments').countDocuments({raw_text: {$exists: true}})
db.getCollection('judgments').countDocuments({raw_text: null})









