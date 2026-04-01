#!/usr/bin/env bash

#
# GIVEN
#
export ENCRYPTION_KEY=my-secret-enc-key
export CONFIG_VALUE=$(cat ./connection-config.json)

#
# WHEN
#
h2o-sonar add config --config-path ./h2o-sonar-config.json --config-type CONNECTION --config-value ${CONFIG_VALUE}  --encryption-key $(ENCRYPTION_KEY)

#
# THEN
#
cat ./h2o-sonar-config.json | grep CONNECTION

# end of test
